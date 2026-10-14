//! Read Out Loud: View ▸ Read this page aloud, Read to end of document, Pause or resume reading
//! and Stop reading, with a reading bar while it runs; File ▸ Save as audio… writes the same
//! speech to a WAV file. Speech comes from the system's engine (`pdfcraft_platform::speech`);
//! the web build has none yet, and says so.
//!
//! Reading runs on a worker thread that extracts one page's text at a time, so a long book
//! starts at once. Pausing stops the engine and remembers the sentence it was on; resuming
//! starts that sentence again.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use pdfcraft_engine::DocId;
use pdfcraft_engine::export::ExportSource;
use pdfcraft_platform::speech::{self, Engine, Speaker, SpeechError, SpeechSettings, Stopper, Voice};

use crate::PdfCraftApp;
use crate::theme::Tokens;

/// The speeds the reading bar offers, as multiples of normal.
const SPEEDS: [f32; 6] = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

/// How much to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    /// The current page only.
    Page,
    /// From the current page to the last.
    ToEnd,
}

/// A sentence: its page and its index among the page's [`speech::speech_chunks`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Position {
    pub page: usize,
    pub chunk: usize,
}

/// What the reading thread reports.
#[derive(Debug, PartialEq)]
enum Event {
    /// Started speaking this sentence.
    At(Position),
    /// This page has no text (a scan, or a picture).
    NoText(usize),
    /// Read to the last page.
    Done,
    Failed(String),
}

struct Reading {
    doc: DocId,
    scope: Scope,
    last: usize,
    at: Position,
    /// Pages found without text.
    silent: usize,
    /// While speaking: stops it. `None` while paused.
    running: Option<(Stopper, mpsc::Receiver<Event>)>,
}

/// Progress of Save as audio: the final message once it is done.
type SaveStatus = Arc<Mutex<Option<String>>>;

#[derive(Default)]
pub(crate) struct ReadAloud {
    /// The voice and speed (saved in the settings).
    pub settings: SpeechSettings,
    /// The system's engine, looked for once.
    engine: Option<Option<Engine>>,
    reading: Option<Reading>,
    voices: Arc<Mutex<Option<Vec<Voice>>>>,
    voices_asked: bool,
    saving: Option<SaveStatus>,
    /// Tests: the engine to use instead of the system's (`Some(None)`: no engine).
    #[cfg(test)]
    engine_override: Option<Option<Engine>>,
}

impl ReadAloud {
    /// The page being read (or paused on) in `doc`.
    pub(crate) fn page_in(&self, doc: DocId) -> Option<usize> {
        self.reading.as_ref().filter(|r| r.doc == doc).map(|r| r.at.page)
    }

    pub(crate) fn is_running(&self) -> bool {
        self.reading.as_ref().is_some_and(|r| r.running.is_some())
    }

    /// The settings to save: `{ "voice": …, "rate": … }`.
    pub(crate) fn persist(&self) -> serde_json::Value {
        serde_json::json!({ "voice": self.settings.voice, "rate": self.settings.rate() })
    }

    /// Restore [`Self::persist`]'s settings, ignoring what doesn't fit.
    pub(crate) fn restore(&mut self, v: &serde_json::Value) {
        if let Some(rate) = v["rate"].as_f64() {
            self.settings.rate = SpeechSettings { rate: rate as f32, ..Default::default() }.rate();
        }
        self.settings.voice = v["voice"].as_str().filter(|s| !s.trim().is_empty() && s.chars().count() <= 256).map(str::to_owned);
    }

    fn stop(&mut self) {
        if let Some(r) = self.reading.take()
            && let Some((stopper, _)) = r.running
        {
            stopper.stop();
        }
    }
}

impl Drop for ReadAloud {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Read `from` to the end of page `last`, one sentence at a time, reporting each on `tx`.
fn read_pages(mut speaker: Speaker, src: ExportSource, from: Position, last: usize, tx: mpsc::Sender<Event>, ctx: Option<egui::Context>) {
    let send = |e: Event| {
        let _ = tx.send(e);
        if let Some(ctx) = &ctx {
            ctx.request_repaint();
        }
    };
    let last = last.min(src.pages.saturating_sub(1));
    let empty = src.pages == 0;
    let mut renderer = pdfcraft_render::PageRenderer::new(src.bytes, src.config);
    for page in (from.page..=last).filter(|_| !empty) {
        let out =
            renderer.render(pdfcraft_render::RenderRequest { page, kind: pdfcraft_render::RequestKind::Text, scale: 1.0, ..Default::default() });
        let chunks = out.text.map(|t| speech::speech_chunks(&t.plain_text())).unwrap_or_default();
        if chunks.is_empty() {
            send(Event::NoText(page));
            continue;
        }
        let skip = if page == from.page { from.chunk } else { 0 };
        for (chunk, text) in chunks.iter().enumerate().skip(skip) {
            send(Event::At(Position { page, chunk }));
            match speaker.speak(text) {
                Ok(()) => {}
                Err(SpeechError::Stopped) => return,
                Err(e) => {
                    send(Event::Failed(e.to_string()));
                    return;
                }
            }
        }
    }
    send(Event::Done);
}

impl PdfCraftApp {
    fn speech_engine(&mut self) -> Option<Engine> {
        #[cfg(test)]
        if let Some(engine) = self.read_aloud.engine_override.clone() {
            return engine;
        }
        self.read_aloud.engine.get_or_insert_with(Engine::detect).clone()
    }

    fn no_speech_engine(&mut self) {
        self.notify(tl!("This system has no speech engine. On Linux and FreeBSD, install espeak-ng.").to_string());
    }

    /// View ▸ Read this page aloud / Read to end of document, from the current page.
    pub(crate) fn read_aloud(&mut self, scope: Scope) {
        let Some((index, doc)) = self.active_ids() else { return };
        let page = self.views.get(index).map_or(0, |v| v.current);
        let pages = self.session.get(doc).map_or(0, |d| d.info.pages.len());
        let last = match scope {
            Scope::Page => page,
            Scope::ToEnd => pages.saturating_sub(1),
        };
        self.start_reading(doc, scope, Position { page, chunk: 0 }, last);
    }

    fn start_reading(&mut self, doc: DocId, scope: Scope, from: Position, last: usize) {
        self.read_aloud.stop();
        let Some(engine) = self.speech_engine() else {
            self.no_speech_engine();
            return;
        };
        let Some(src) = self.session.get(doc).map(pdfcraft_engine::Document::export_source) else { return };
        let speaker = match Speaker::new(&engine, &self.read_aloud.settings) {
            Ok(s) => s,
            Err(e) => {
                self.notify_fmt("Read Out Loud stopped: {e}", &[("e", &e.to_string())]);
                return;
            }
        };
        let stopper = speaker.stopper();
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        let spawned = std::thread::Builder::new().name("pdfcraft-read-aloud".into()).spawn(move || read_pages(speaker, src, from, last, tx, ctx));
        if let Err(e) = spawned {
            self.notify_fmt("Read Out Loud stopped: {e}", &[("e", &e.to_string())]);
            return;
        }
        self.read_aloud.reading = Some(Reading { doc, scope, last, at: from, silent: 0, running: Some((stopper, rx)) });
    }

    /// View ▸ Pause or resume reading.
    pub(crate) fn read_aloud_pause(&mut self) {
        let Some(r) = self.read_aloud.reading.as_mut() else { return };
        match r.running.take() {
            Some((stopper, _)) => stopper.stop(),
            None => {
                let (doc, scope, at, last) = (r.doc, r.scope, r.at, r.last);
                self.start_reading(doc, scope, at, last);
            }
        }
    }

    /// View ▸ Stop reading.
    pub(crate) fn read_aloud_stop(&mut self) {
        self.read_aloud.stop();
    }

    /// Start again from the current sentence, after the voice or speed changed.
    fn restart_reading(&mut self) {
        if let Some(r) = self.read_aloud.reading.as_ref().filter(|r| r.running.is_some()) {
            let (doc, scope, at, last) = (r.doc, r.scope, r.at, r.last);
            self.start_reading(doc, scope, at, last);
        }
    }

    /// Follow the reading: turn to the page being read, and report the end.
    pub(crate) fn poll_read_aloud(&mut self) {
        self.poll_audio_save();
        let Some(r) = self.read_aloud.reading.as_mut() else { return };
        if self.session.get(r.doc).is_none() {
            self.read_aloud.stop();
            return;
        }
        let Some((_, rx)) = r.running.as_ref() else { return };
        let before = r.at.page;
        let mut end: Option<Option<String>> = None;
        loop {
            match rx.try_recv() {
                Ok(Event::At(p)) => r.at = p,
                Ok(Event::NoText(_)) => r.silent += 1,
                Ok(Event::Done) => {
                    end = Some(None);
                    break;
                }
                Ok(Event::Failed(e)) => {
                    end = Some(Some(e));
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    end = Some(Some("the reader stopped unexpectedly".into()));
                    break;
                }
            }
        }
        let (doc, page, scope, silent) = (r.doc, r.at.page, r.scope, r.silent);
        if page != before
            && let Some(v) = self.views.iter_mut().find(|v| v.id == doc)
            && v.current != page
        {
            v.goto = Some((page, 0.0));
        }
        match end {
            None => {}
            Some(None) => {
                self.read_aloud.reading = None;
                if scope == Scope::Page && silent > 0 {
                    self.notify(tl!("This page has no text to read. Recognize text (OCR) first.").to_string());
                }
            }
            Some(Some(e)) => {
                self.read_aloud.reading = None;
                self.notify_fmt("Read Out Loud stopped: {e}", &[("e", &e)]);
            }
        }
    }

    /// The reading bar, at the foot of the window while the active document is read or paused.
    pub(crate) fn read_aloud_bar(&mut self, ctx: &egui::Context) {
        let Some((_, doc)) = self.active_ids() else { return };
        let Some(page) = self.read_aloud.page_in(doc) else { return };
        let total = self.session.get(doc).map_or(0, |d| d.info.pages.len());
        let running = self.read_aloud.is_running();
        self.ask_for_voices();
        let t = Tokens::get(ctx);
        let (mut toggle, mut stop, mut restart) = (false, false, false);
        egui::Area::new(egui::Id::new("read-aloud-bar")).anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -28.0)).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (icon, tip) = if running { ("pause", tl!("Pause ({key})")) } else { ("play", tl!("Resume ({key})")) };
                    let tip = crate::commands::command_tip(ui.ctx(), tip, "view.read_aloud.pause");
                    toggle = crate::icons::button(ui, icon, 30.0, false, &tip).clicked();
                    let tip = crate::commands::command_tip(ui.ctx(), tl!("Stop reading ({key})"), "view.read_aloud.stop");
                    stop = crate::icons::button(ui, "square", 30.0, false, &tip).clicked();
                    let (n, total) = ((page + 1).to_string(), total.to_string());
                    let status = if running { tl!("Reading page {n} of {total}") } else { tl!("Paused on page {n} of {total}") };
                    ui.label(egui::RichText::new(crate::i18n::fmt(status, &[("n", &n), ("total", &total)])).color(t.text));
                    ui.separator();
                    ui.label(egui::RichText::new(tl!("Speed")).color(t.text_faint));
                    let rate = self.read_aloud.settings.rate();
                    egui::ComboBox::from_id_salt("read-aloud-speed").selected_text(speed_label(rate)).show_ui(ui, |ui| {
                        for s in SPEEDS {
                            if ui.selectable_label((s - rate).abs() < 0.01, speed_label(s)).clicked() {
                                self.read_aloud.settings.rate = s;
                                restart = true;
                            }
                        }
                    });
                    ui.label(egui::RichText::new(tl!("Voice")).color(t.text_faint));
                    let voices = self.read_aloud.voices.lock().ok().and_then(|v| v.clone()).unwrap_or_default();
                    let current = self.read_aloud.settings.voice.clone();
                    let shown = current
                        .as_deref()
                        .map(|id| voices.iter().find(|v| v.id == id).map_or(id, |v| v.name.as_str()).to_string())
                        .unwrap_or_else(|| tl!("System voice").to_string());
                    egui::ComboBox::from_id_salt("read-aloud-voice").selected_text(shown).width(180.0).show_ui(ui, |ui| {
                        if ui.selectable_label(current.is_none(), tl!("System voice")).clicked() && current.is_some() {
                            self.read_aloud.settings.voice = None;
                            restart = true;
                        }
                        for v in &voices {
                            let label = if v.language.is_empty() { v.name.clone() } else { format!("{} · {}", v.name, v.language) };
                            if ui.selectable_label(current.as_deref() == Some(v.id.as_str()), label).clicked() && current.as_deref() != Some(&v.id) {
                                self.read_aloud.settings.voice = Some(v.id.clone());
                                restart = true;
                            }
                        }
                    });
                });
            });
        });
        if toggle {
            self.read_aloud_pause();
        } else if stop {
            self.read_aloud_stop();
        } else if restart {
            self.restart_reading();
        }
    }

    /// List the voices once, off the interface thread (Windows takes a second).
    fn ask_for_voices(&mut self) {
        if self.read_aloud.voices_asked {
            return;
        }
        self.read_aloud.voices_asked = true;
        let Some(engine) = self.speech_engine() else { return };
        let into = self.read_aloud.voices.clone();
        let ctx = self.ctx.clone();
        let _ = std::thread::Builder::new().name("pdfcraft-voices".into()).spawn(move || {
            let list = speech::voices(&engine).unwrap_or_default();
            if let Ok(mut v) = into.lock() {
                *v = Some(list);
            }
            if let Some(ctx) = ctx {
                ctx.request_repaint();
            }
        });
    }

    /// File ▸ Save as audio…: read the whole document into a WAV file.
    pub(crate) fn save_as_audio(&mut self) {
        if !self.commit_form_typing() {
            return;
        }
        let Some((_, id)) = self.active_ids() else { return };
        if !self.may_copy(id) {
            return;
        }
        if self.read_aloud.saving.is_some() {
            self.notify(tl!("Saving audio…").to_string());
            return;
        }
        let Some(engine) = self.speech_engine() else {
            self.no_speech_engine();
            return;
        };
        if !engine.can_record() {
            self.notify(tl!("This speech engine can't save audio. Install espeak-ng.").to_string());
            return;
        }
        let Some(doc) = self.session.get(id) else { return };
        let src = doc.export_source();
        let name = format!("{}.wav", doc.name.trim_end_matches(".pdf").trim_end_matches(".PDF"));
        let settings = self.read_aloud.settings.clone();
        let start = move |app: &mut Self, path: std::path::PathBuf, inline: bool| {
            let status: SaveStatus = Arc::new(Mutex::new(None));
            let into = status.clone();
            let lang = crate::i18n::current();
            let ctx = app.ctx.clone();
            let work = move || {
                crate::i18n::set_current(lang);
                let msg = save_audio(&engine, &settings, src, &path);
                if let Ok(mut s) = into.lock() {
                    *s = Some(msg);
                }
                if let Some(ctx) = ctx {
                    ctx.request_repaint();
                }
            };
            app.read_aloud.saving = Some(status);
            app.notify(tl!("Saving audio…").to_string());
            if inline {
                work(); // tests: synchronous
            } else if std::thread::Builder::new().name("pdfcraft-save-audio".into()).spawn(work).is_err() {
                app.read_aloud.saving = None;
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        match self.save_override.clone() {
            Some(p) => start(self, p.into(), true),
            None => {
                let dialog = rfd::AsyncFileDialog::new().add_filter("WAV", &["wav"]).set_file_name(&name);
                self.ask_one(crate::pickers::Ask::Save(dialog), None, move |app, p| start(app, p, false));
            }
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (start, name);
    }

    fn poll_audio_save(&mut self) {
        let Some(status) = self.read_aloud.saving.clone() else { return };
        let done = status.lock().ok().and_then(|mut s| s.take());
        if let Some(msg) = done {
            self.read_aloud.saving = None;
            self.notify(msg);
        }
    }
}

/// "1.25×".
fn speed_label(rate: f32) -> String {
    format!("{}×", (rate * 100.0).round() / 100.0)
}

/// Speak every page of `src` into the WAV file `path` (staged beside it, then renamed over it),
/// returning the message to show.
fn save_audio(engine: &Engine, settings: &SpeechSettings, src: ExportSource, path: &std::path::Path) -> String {
    let pages = src.pages;
    let mut ex = pdfcraft_engine::export::Exporter::from_source(src);
    let chunks: Vec<String> = (0..pages).flat_map(|p| ex.text(p).map(|t| speech::speech_chunks(&t)).unwrap_or_default()).collect();
    if chunks.is_empty() {
        return tl!("None of these pages has text to read. Recognize text (OCR) first.").to_string();
    }
    let fail = |e: &dyn std::fmt::Display| crate::i18n::fmt(tl!("Couldn't save audio: {e}"), &[("e", &e.to_string())]);
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "audio.wav".into());
    let (tmp, file) = match pdfcraft_platform::staging::create_staging(
        dir,
        &name,
        pdfcraft_platform::staging::StagingName::TagThenSuffix,
        pdfcraft_platform::staging::staging_suffixes(),
    ) {
        Ok(staged) => staged,
        Err(e) => return fail(&e),
    };
    // Closed, so the engine (another process) can write it on Windows.
    drop(file);
    let result = speech::record(engine, settings, &chunks, &tmp, &Stopper::new())
        .map_err(|e| e.to_string())
        .and_then(|()| std::fs::rename(&tmp, path).map_err(|e| e.to_string()));
    match result {
        Ok(()) => crate::i18n::fmt(tl!("Saved audio to {path}"), &[("path", &path.display().to_string())]),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            fail(&e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speeds_are_labelled_without_trailing_zeros() {
        assert_eq!(speed_label(1.0), "1×");
        assert_eq!(speed_label(1.25), "1.25×");
        assert_eq!(speed_label(0.75), "0.75×");
    }

    #[test]
    fn saved_settings_are_restored_within_range() {
        let mut r = ReadAloud::default();
        r.restore(&serde_json::json!({ "voice": "Microsoft Zira Desktop", "rate": 1.5 }));
        assert_eq!(r.settings.voice.as_deref(), Some("Microsoft Zira Desktop"));
        assert_eq!(r.settings.rate, 1.5);
        let mut back = ReadAloud::default();
        back.restore(&r.persist());
        assert_eq!(back.settings, r.settings);
        // Junk is ignored or brought into range, never trusted.
        let mut r = ReadAloud::default();
        r.restore(&serde_json::json!({ "voice": "x".repeat(300), "rate": 1e30 }));
        assert_eq!((r.settings.voice.as_deref(), r.settings.rate), (None, speech::MAX_RATE));
        let mut r = ReadAloud::default();
        r.restore(&serde_json::json!({ "voice": 7, "rate": "fast" }));
        assert_eq!(r.settings, SpeechSettings::default());
        r.restore(&serde_json::Value::Null);
        assert_eq!(r.settings, SpeechSettings::default());
    }

    /// An engine whose program doesn't exist: anything it is asked to say fails to start.
    fn missing_engine() -> Engine {
        Engine::new(speech::EngineKind::Espeak, "pdfcraft-no-such-speech-program")
    }

    /// One page per entry showing that text (an empty entry is a blank page, like a scan).
    fn pages_showing(texts: &[&str]) -> Vec<u8> {
        let n = texts.len();
        let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 4 + 2 * i)).collect();
        let mut objs = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!("<< /Type /Pages /Kids [{}] /Count {n} /MediaBox [0 0 200 300] >>", kids.join(" ")),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        for (i, text) in texts.iter().enumerate() {
            objs.push(format!("<< /Type /Page /Parent 2 0 R /Contents {} 0 R /Resources << /Font << /F1 3 0 R >> >> >>", 5 + 2 * i));
            let body = if text.is_empty() { String::new() } else { format!("BT /F1 8 Tf 5 150 Td ({text}) Tj ET") };
            objs.push(format!("<< /Length {} >>\nstream\n{body}\nendstream", body.len()));
        }
        let mut out = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
        for o in offsets {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).as_bytes());
        out
    }

    fn app_with(texts: &[&str], engine: Option<Engine>) -> PdfCraftApp {
        let mut app = PdfCraftApp::new();
        app.read_aloud.engine_override = Some(engine);
        app.open_bytes("book.pdf", None, pages_showing(texts)).unwrap();
        app
    }

    /// Poll until the reading ends (the worker runs on its own thread).
    fn finish(app: &mut PdfCraftApp) {
        let start = std::time::Instant::now();
        while app.read_aloud.reading.is_some() {
            assert!(start.elapsed() < std::time::Duration::from_secs(20), "reading never ended");
            app.poll_read_aloud();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn toast(app: &PdfCraftApp) -> String {
        app.toast.as_ref().map(|(m, _)| m.clone()).unwrap_or_default()
    }

    #[test]
    fn without_an_engine_reading_and_saving_say_so_and_start_nothing() {
        let mut app = app_with(&["Hello."], None);
        for id in ["view.read_aloud.page", "view.read_aloud.to_end", "export.audio"] {
            app.toast = None;
            assert!(app.execute(id), "{id}");
            assert!(toast(&app).contains("no speech engine"), "{id}: {}", toast(&app));
            assert!(app.read_aloud.reading.is_none() && app.read_aloud.saving.is_none(), "{id}");
        }
        // Pause and stop with nothing being read do nothing.
        assert!(app.execute("view.read_aloud.pause") && app.execute("view.read_aloud.stop"));
        assert!(app.read_aloud.reading.is_none());
    }

    #[test]
    fn a_page_without_text_is_reported() {
        let mut app = app_with(&[""], Some(missing_engine()));
        app.execute("view.read_aloud.page");
        assert!(app.read_aloud.is_running());
        finish(&mut app);
        assert_eq!(toast(&app), "This page has no text to read. Recognize text (OCR) first.");
    }

    #[test]
    fn reading_to_the_end_turns_to_the_page_being_read_and_reports_a_failing_engine() {
        let mut app = app_with(&["", "Chapter one. It begins."], Some(missing_engine()));
        app.execute("view.read_aloud.to_end");
        finish(&mut app);
        // The blank first page is skipped; the engine fails on page 2's first sentence.
        assert_eq!(app.views[0].goto, Some((1, 0.0)), "the view follows the reading");
        assert!(toast(&app).starts_with("Read Out Loud stopped: the speech engine didn't start"), "{}", toast(&app));
    }

    #[test]
    fn closing_the_document_stops_the_reading() {
        let mut app = app_with(&["", ""], Some(missing_engine()));
        app.execute("view.read_aloud.to_end");
        let doc = app.views[0].id;
        app.session.close(doc);
        app.poll_read_aloud();
        assert!(app.read_aloud.reading.is_none());
    }

    #[test]
    fn reading_an_empty_document_finishes_without_speaking() {
        let (tx, rx) = mpsc::channel();
        let speaker = Speaker::new(&missing_engine(), &SpeechSettings::default()).unwrap();
        let src = ExportSource { bytes: Arc::new(Vec::new()), config: Default::default(), pages: 0, sizes: Vec::new() };
        read_pages(speaker, src, Position::default(), 5, tx, None);
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), [Event::Done]);
    }

    #[test]
    fn saving_audio_without_text_writes_nothing() {
        let dir = std::env::temp_dir().join(format!("pdfcraft-save-audio-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut app = app_with(&["", ""], Some(missing_engine()));
        app.save_override = Some(dir.join("book.wav").to_string_lossy().into_owned());
        app.execute("export.audio");
        app.poll_read_aloud();
        assert_eq!(toast(&app), "None of these pages has text to read. Recognize text (OCR) first.");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        // With text, a failing engine leaves no file behind either.
        let mut app = app_with(&["Some words."], Some(missing_engine()));
        app.save_override = Some(dir.join("book.wav").to_string_lossy().into_owned());
        app.execute("export.audio");
        app.poll_read_aloud();
        assert!(toast(&app).starts_with("Couldn't save audio:"), "{}", toast(&app));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "the staging file was removed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
