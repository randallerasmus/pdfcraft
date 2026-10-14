//! Read Out Loud: text to speech through the operating system's own synthesizer.
//!
//! Each platform already ships a speech engine, so PdfCraft drives it as a child process
//! instead of linking a speech library: Windows' System.Speech through PowerShell, macOS's
//! `say`, and eSpeak NG or Speech Dispatcher on Linux and FreeBSD. Text always travels on the
//! child's standard input (base64 on Windows) or as a single argument after `--`, never through a
//! shell, so a document's text can't become a command. The web build has no engine yet.
//!
//! [`Speaker`] speaks one chunk at a time and returns when it has been heard, so a caller can
//! follow the reading position; [`record`] writes the speech to a WAV file instead. Either is
//! interrupted at once by its [`Stopper`]. [`speech_chunks`] cuts page text into sentence-sized
//! chunks, which is also how far a pause rewinds.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

/// The longest chunk [`speech_chunks`] returns, in characters. A longer sentence is cut between
/// words, so a pause never rewinds far and no command line or line read grows without bound.
pub const MAX_CHUNK_CHARS: usize = 400;
/// The slowest and fastest reading speeds, as multiples of the engine's normal speed.
pub const MIN_RATE: f32 = 0.5;
pub const MAX_RATE: f32 = 3.0;
/// The longest voice name accepted; real ones are a few dozen characters.
const MAX_VOICE_CHARS: usize = 256;
/// At most this many voices are listed, however many lines an engine prints.
const MAX_VOICES: usize = 512;
/// How often a waiting call looks for the child's exit or a stop.
const POLL: Duration = Duration::from_millis(15);

/// Which synthesizer an [`Engine`] drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// System.Speech (SAPI), through Windows PowerShell.
    Windows,
    /// macOS's `say`.
    Say,
    /// `espeak-ng` (or the older `espeak`).
    Espeak,
    /// Speech Dispatcher's `spd-say`, which can speak but not record.
    SpeechDispatcher,
}

/// A speech engine found on this computer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Engine {
    kind: EngineKind,
    program: PathBuf,
}

impl Engine {
    /// The engine to use here, or `None` when the system has none PdfCraft can drive.
    pub fn detect() -> Option<Engine> {
        detect()
    }

    /// `kind`'s engine run from `program`, for tests and for packagers who ship their own.
    pub fn new(kind: EngineKind, program: impl Into<PathBuf>) -> Engine {
        Engine { kind, program: program.into() }
    }

    pub fn kind(&self) -> EngineKind {
        self.kind
    }

    /// The engine's name, for the interface and the automation tools.
    pub fn name(&self) -> &'static str {
        match self.kind {
            EngineKind::Windows => "Windows speech",
            EngineKind::Say => "macOS speech",
            EngineKind::Espeak => "eSpeak NG",
            EngineKind::SpeechDispatcher => "Speech Dispatcher",
        }
    }

    /// Whether [`record`] can write this engine's speech to a file.
    pub fn can_record(&self) -> bool {
        self.kind != EngineKind::SpeechDispatcher
    }
}

#[cfg(target_arch = "wasm32")]
fn detect() -> Option<Engine> {
    None
}

#[cfg(all(windows, not(target_arch = "wasm32")))]
fn detect() -> Option<Engine> {
    // The full path, so a powershell.exe elsewhere on PATH (or in the working folder) is never run.
    let root = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let program = root.join(r"System32\WindowsPowerShell\v1.0\powershell.exe");
    program.is_file().then_some(Engine { kind: EngineKind::Windows, program })
}

#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
fn detect() -> Option<Engine> {
    let program = PathBuf::from("/usr/bin/say");
    program.is_file().then_some(Engine { kind: EngineKind::Say, program })
}

#[cfg(not(any(windows, target_os = "macos", target_arch = "wasm32")))]
fn detect() -> Option<Engine> {
    let path = std::env::var_os("PATH")?;
    for (name, kind) in [("espeak-ng", EngineKind::Espeak), ("espeak", EngineKind::Espeak), ("spd-say", EngineKind::SpeechDispatcher)] {
        // Only absolute PATH entries: a relative one would find a program planted in the
        // working folder.
        if let Some(program) = std::env::split_paths(&path).filter(|d| d.is_absolute()).map(|d| d.join(name)).find(|p| p.is_file()) {
            return Some(Engine { kind, program });
        }
    }
    None
}

/// How to read: the voice, the speed and the volume.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechSettings {
    /// A voice id from [`voices`]; `None` is the system's default voice.
    pub voice: Option<String>,
    /// A multiple of the normal speed, kept within [`MIN_RATE`]–[`MAX_RATE`].
    pub rate: f32,
    /// 0–100.
    pub volume: u8,
}

impl Default for SpeechSettings {
    fn default() -> Self {
        SpeechSettings { voice: None, rate: 1.0, volume: 100 }
    }
}

impl SpeechSettings {
    /// The speed to use: within range, and normal for NaN or infinity.
    pub fn rate(&self) -> f32 {
        if self.rate.is_finite() { self.rate.clamp(MIN_RATE, MAX_RATE) } else { 1.0 }
    }

    fn volume(&self) -> u8 {
        self.volume.min(100)
    }

    /// The voice to ask for, refusing a name that an engine could read as an option or that
    /// carries control characters (settings files are untrusted).
    fn voice(&self) -> Result<Option<&str>, SpeechError> {
        match self.voice.as_deref().map(str::trim) {
            None | Some("") => Ok(None),
            Some(v) if v.starts_with('-') || v.chars().any(char::is_control) || v.chars().count() > MAX_VOICE_CHARS => Err(SpeechError::InvalidVoice),
            Some(v) => Ok(Some(v)),
        }
    }
}

/// SAPI's rate, −10 to 10, where each step is about 3^(1/10) times faster.
fn sapi_rate(rate: f32) -> i32 {
    ((rate.ln() / 3f32.ln()) * 10.0).round().clamp(-10.0, 10.0) as i32
}

/// Words per minute for `say` and eSpeak, whose normal speed is 175.
fn words_per_minute(rate: f32) -> u32 {
    (175.0 * rate).round().clamp(80.0, 525.0) as u32
}

/// Speech Dispatcher's −100 to 100 scales, 0 being normal.
fn dispatcher_rate(rate: f32) -> i32 {
    ((rate - 1.0) * 100.0).round().clamp(-100.0, 100.0) as i32
}

fn dispatcher_volume(volume: u8) -> i32 {
    i32::from(volume.min(100)) * 2 - 100
}

/// A voice an engine offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Voice {
    /// What to put in [`SpeechSettings::voice`].
    pub id: String,
    /// What to show.
    pub name: String,
    /// The language tag the engine gives, possibly empty.
    pub language: String,
}

/// Why speech didn't happen.
#[derive(Debug)]
pub enum SpeechError {
    /// This system has no speech engine PdfCraft can use.
    NoEngine,
    /// This engine can't do what was asked (Speech Dispatcher can't record).
    CannotRecord,
    /// The voice setting isn't a voice name.
    InvalidVoice,
    /// The engine didn't start.
    Start(std::io::Error),
    /// The engine stopped with an error.
    Failed(String),
    /// [`Stopper::stop`] was called.
    Stopped,
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeechError::NoEngine => write!(f, "this system has no speech engine (on Linux and FreeBSD, install espeak-ng or speech-dispatcher)"),
            SpeechError::CannotRecord => write!(f, "Speech Dispatcher can read aloud but can't save audio (install espeak-ng)"),
            SpeechError::InvalidVoice => write!(f, "the voice setting isn't a voice name"),
            SpeechError::Start(e) => write!(f, "the speech engine didn't start: {e}"),
            SpeechError::Failed(m) => write!(f, "the speech engine stopped: {m}"),
            SpeechError::Stopped => write!(f, "reading was stopped"),
        }
    }
}

impl std::error::Error for SpeechError {}

/// Stops a [`Speaker`] or a [`record`] from another thread, at once: the child process is
/// ended and the blocked call returns [`SpeechError::Stopped`]. Cloning shares the stop.
#[derive(Clone, Default)]
pub struct Stopper(Arc<StopState>);

#[derive(Default)]
struct StopState {
    stopped: AtomicBool,
    child: Mutex<Option<Child>>,
}

impl Stopper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stop(&self) {
        self.0.stopped.store(true, Ordering::SeqCst);
        if let Some(child) = self.child().as_mut() {
            let _ = child.kill();
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.0.stopped.load(Ordering::SeqCst)
    }

    /// The running child. A thread that panicked while holding the lock left nothing half
    /// changed (it only stores or takes the child), so a poisoned lock is used as it is.
    fn child(&self) -> MutexGuard<'_, Option<Child>> {
        self.0.child.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Hold `child` so [`Self::stop`] can end it; a stop that came first ends it now.
    fn adopt(&self, mut child: Child) -> Result<(), SpeechError> {
        if self.is_stopped() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(SpeechError::Stopped);
        }
        *self.child() = Some(child);
        Ok(())
    }

    /// Wait for the held child to exit, ending it when stopped.
    fn wait(&self) -> Result<ExitStatus, SpeechError> {
        loop {
            {
                let mut guard = self.child();
                let Some(child) = guard.as_mut() else { return Err(SpeechError::Stopped) };
                if self.is_stopped() {
                    let _ = child.kill();
                    let _ = child.wait();
                    *guard = None;
                    return Err(SpeechError::Stopped);
                }
                match child.try_wait() {
                    Ok(Some(status)) => {
                        *guard = None;
                        return Ok(status);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        let _ = child.kill();
                        *guard = None;
                        return Err(SpeechError::Failed(e.to_string()));
                    }
                }
            }
            std::thread::sleep(POLL);
        }
    }

    /// End and reap the held child, if any.
    fn end(&self) {
        if let Some(mut child) = self.child().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Reads aloud, one chunk per [`Speaker::speak`] call.
pub struct Speaker {
    engine: Engine,
    settings: SpeechSettings,
    stopper: Stopper,
    /// Windows keeps one PowerShell running between chunks, so there's no start-up pause
    /// between sentences: its input and output.
    windows: Option<(ChildStdin, BufReader<ChildStdout>)>,
}

impl Speaker {
    pub fn new(engine: &Engine, settings: &SpeechSettings) -> Result<Speaker, SpeechError> {
        settings.voice()?;
        Ok(Speaker { engine: engine.clone(), settings: settings.clone(), stopper: Stopper::new(), windows: None })
    }

    /// Stops this speaker from another thread.
    pub fn stopper(&self) -> Stopper {
        self.stopper.clone()
    }

    /// Say `text` and return once it has been spoken.
    pub fn speak(&mut self, text: &str) -> Result<(), SpeechError> {
        if self.stopper.is_stopped() {
            return Err(SpeechError::Stopped);
        }
        let text = one_line(text);
        if text.is_empty() {
            return Ok(());
        }
        match self.engine.kind {
            EngineKind::Windows => self.speak_windows(&text),
            _ => {
                let mut command = speak_command(&self.engine, &self.settings, None, &text)?;
                let stdin = self.engine.kind != EngineKind::SpeechDispatcher;
                command.stdin(if stdin { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::null()).stderr(Stdio::null());
                let mut child = command.spawn().map_err(SpeechError::Start)?;
                let input = child.stdin.take();
                self.stopper.adopt(child)?;
                if let Some(mut input) = input {
                    if input.write_all(text.as_bytes()).is_err() && self.stopper.is_stopped() {
                        return Err(SpeechError::Stopped);
                    }
                }
                exit_ok(&self.engine, self.stopper.wait()?)
            }
        }
    }

    fn speak_windows(&mut self, text: &str) -> Result<(), SpeechError> {
        if self.windows.is_none() {
            let mut command = speak_command(&self.engine, &self.settings, None, text)?;
            command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
            let mut child = command.spawn().map_err(SpeechError::Start)?;
            let (Some(input), Some(output)) = (child.stdin.take(), child.stdout.take()) else {
                let _ = child.kill();
                let _ = child.wait();
                return Err(SpeechError::Failed("no pipe to the speech engine".into()));
            };
            self.stopper.adopt(child)?;
            let mut output = BufReader::new(output);
            // "ready" once the voice is set up; anything else means the script failed (an
            // unknown voice, say).
            self.expect_line(&mut output, "ready")?;
            self.windows = Some((input, output));
        }
        let Some((input, output)) = self.windows.as_mut() else { return Err(SpeechError::Stopped) };
        let line = format!("{}\n", base64(text.as_bytes()));
        if input.write_all(line.as_bytes()).and_then(|()| input.flush()).is_err() {
            return Err(self.lost());
        }
        let mut reply = String::new();
        match output.by_ref().take(64).read_line(&mut reply) {
            Ok(n) if n > 0 && reply.trim() == "done" => Ok(()),
            _ => Err(self.lost()),
        }
    }

    fn expect_line(&mut self, output: &mut BufReader<ChildStdout>, want: &str) -> Result<(), SpeechError> {
        let mut line = String::new();
        match output.by_ref().take(64).read_line(&mut line) {
            Ok(n) if n > 0 && line.trim() == want => Ok(()),
            _ => Err(self.lost()),
        }
    }

    /// The error for a PowerShell that went away: a stop, or a failure (most often a voice
    /// that isn't installed).
    fn lost(&mut self) -> SpeechError {
        self.windows = None;
        self.stopper.end();
        if self.stopper.is_stopped() { SpeechError::Stopped } else { SpeechError::Failed("Windows speech couldn't read with this voice".into()) }
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        // Closing PowerShell's input ends its loop; ending it as well covers a voice still talking.
        self.windows = None;
        self.stopper.end();
    }
}

/// Speak `chunks` into the WAV file `out` (created or replaced; the caller stages it). Blocks
/// until the file is complete or `stopper` stops it.
pub fn record(engine: &Engine, settings: &SpeechSettings, chunks: &[String], out: &Path, stopper: &Stopper) -> Result<(), SpeechError> {
    if !engine.can_record() {
        return Err(SpeechError::CannotRecord);
    }
    let windows = engine.kind == EngineKind::Windows;
    let mut command = speak_command(engine, settings, Some(out), "")?;
    command.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = command.spawn().map_err(SpeechError::Start)?;
    let input = child.stdin.take();
    stopper.adopt(child)?;
    if let Some(mut input) = input {
        for chunk in chunks {
            let line = one_line(chunk);
            if line.is_empty() {
                continue;
            }
            let line = if windows { base64(line.as_bytes()) } else { line };
            if input.write_all(line.as_bytes()).and_then(|()| input.write_all(b"\n")).is_err() {
                // The engine went away; its exit status says why.
                break;
            }
        }
        // Dropped here: end of input, so the engine finishes the file and exits.
    }
    exit_ok(engine, stopper.wait()?)
}

fn exit_ok(engine: &Engine, status: ExitStatus) -> Result<(), SpeechError> {
    if status.success() { Ok(()) } else { Err(SpeechError::Failed(format!("{} exited with {status}", engine.name()))) }
}

/// The voices `engine` offers. Runs the engine once, so call it off the interface thread.
pub fn voices(engine: &Engine) -> Result<Vec<Voice>, SpeechError> {
    let mut command = Command::new(&engine.program);
    match engine.kind {
        EngineKind::Windows => {
            windows_command(&mut command, LIST_VOICES_SCRIPT);
        }
        EngineKind::Say => {
            command.args(["-v", "?"]);
        }
        EngineKind::Espeak => {
            command.arg("--voices");
        }
        EngineKind::SpeechDispatcher => {
            command.arg("-L");
        }
    }
    command.stdin(Stdio::null()).stderr(Stdio::null());
    let output = command.output().map_err(SpeechError::Start)?;
    if !output.status.success() {
        return Err(SpeechError::Failed(format!("{} exited with {}", engine.name(), output.status)));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut list = match engine.kind {
        EngineKind::Windows => parse_windows_voices(&text),
        EngineKind::Say => parse_say_voices(&text),
        EngineKind::Espeak => parse_espeak_voices(&text),
        EngineKind::SpeechDispatcher => parse_dispatcher_voices(&text),
    };
    list.truncate(MAX_VOICES);
    Ok(list)
}

/// The command that speaks (to the speakers, or into `out`). Text goes on standard input,
/// except for Speech Dispatcher, which takes `text` as its last argument.
fn speak_command(engine: &Engine, settings: &SpeechSettings, out: Option<&Path>, text: &str) -> Result<Command, SpeechError> {
    let voice = settings.voice()?;
    let rate = settings.rate();
    let mut command = Command::new(&engine.program);
    match engine.kind {
        EngineKind::Windows => {
            windows_command(&mut command, SPEAK_SCRIPT);
            // Settings travel as environment variables, which PowerShell never parses as code.
            command.env("PDFCRAFT_SPEECH_RATE", sapi_rate(rate).to_string());
            command.env("PDFCRAFT_SPEECH_VOLUME", settings.volume().to_string());
            command.env("PDFCRAFT_SPEECH_VOICE", voice.unwrap_or(""));
            command.env("PDFCRAFT_SPEECH_OUT", out.map(Path::as_os_str).unwrap_or_default());
        }
        EngineKind::Say => {
            if let Some(v) = voice {
                command.arg("-v").arg(v);
            }
            command.arg("-r").arg(words_per_minute(rate).to_string());
            if let Some(out) = out {
                command.args(["--file-format=WAVE", "--data-format=LEI16@22050", "-o"]).arg(out);
            }
        }
        EngineKind::Espeak => {
            if let Some(v) = voice {
                command.arg("-v").arg(v);
            }
            command.arg("-s").arg(words_per_minute(rate).to_string());
            command.arg("-a").arg((u32::from(settings.volume()) * 2).to_string());
            if let Some(out) = out {
                command.arg("-w").arg(out);
            }
            command.arg("--stdin");
        }
        EngineKind::SpeechDispatcher => {
            if let Some(v) = voice {
                command.arg("-y").arg(v);
            }
            command.arg("-r").arg(dispatcher_rate(rate).to_string());
            command.arg("-i").arg(dispatcher_volume(settings.volume()).to_string());
            command.arg("-w").arg("--").arg(text);
        }
    }
    Ok(command)
}

/// Reads one base64 line of UTF-8 text at a time and speaks it, saying "done" after each, so
/// the caller knows where the reading is. Writes "ready" once the voice is set up.
const SPEAK_SCRIPT: &str = r#"$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:PDFCRAFT_SPEECH_VOICE) { $s.SelectVoice($env:PDFCRAFT_SPEECH_VOICE) }
$s.Rate = [int]$env:PDFCRAFT_SPEECH_RATE
$s.Volume = [int]$env:PDFCRAFT_SPEECH_VOLUME
if ($env:PDFCRAFT_SPEECH_OUT) { $s.SetOutputToWaveFile($env:PDFCRAFT_SPEECH_OUT) } else { $s.SetOutputToDefaultAudioDevice() }
[Console]::Out.WriteLine('ready')
while ($null -ne ($l = [Console]::In.ReadLine())) {
  $s.Speak([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($l)))
  [Console]::Out.WriteLine('done')
}
$s.Dispose()
"#;

/// One voice per line: name, a tab, and its language tag.
const LIST_VOICES_SCRIPT: &str = r#"$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$out = [Console]::OpenStandardOutput()
foreach ($v in $s.GetInstalledVoices()) {
  if ($v.Enabled) {
    $b = [Text.Encoding]::UTF8.GetBytes($v.VoiceInfo.Name + "`t" + $v.VoiceInfo.Culture.Name + "`n")
    $out.Write($b, 0, $b.Length)
  }
}
$out.Flush()
"#;

/// Run `script` in Windows PowerShell without a profile, a prompt or a console window. It goes
/// as `-EncodedCommand` (base64 of UTF-16LE), which needs no quoting.
fn windows_command(command: &mut Command, script: &str) {
    let utf16: Vec<u8> = script.encode_utf16().flat_map(u16::to_le_bytes).collect();
    command.args(["-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand"]).arg(base64(&utf16));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: the desktop app has no console, and a new one would flash up.
        command.creation_flags(0x0800_0000);
    }
}

fn parse_windows_voices(text: &str) -> Vec<Voice> {
    text.lines()
        .filter_map(|line| {
            let (name, language) = line.split_once('\t').unwrap_or((line, ""));
            let name = name.trim();
            (!name.is_empty()).then(|| Voice { id: name.into(), name: name.into(), language: language.trim().into() })
        })
        .collect()
}

/// `say -v ?` lines: the name (which may contain spaces), the locale, then `# sample`.
fn parse_say_voices(text: &str) -> Vec<Voice> {
    text.lines()
        .filter_map(|line| {
            let head = line.split_once('#').map_or(line, |(h, _)| h).trim_end();
            let (name, language) = head.rsplit_once(char::is_whitespace)?;
            let name = name.trim();
            (!name.is_empty()).then(|| Voice { id: name.into(), name: name.into(), language: language.into() })
        })
        .collect()
}

/// `espeak-ng --voices`: a header, then `Pty Language Age/Gender VoiceName File Other…`.
/// The language is what `-v` takes.
fn parse_espeak_voices(text: &str) -> Vec<Voice> {
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let language = words.nth(1)?;
            let name = words.nth(1)?;
            Some(Voice { id: language.into(), name: format!("{} ({language})", name.replace('_', " ")), language: language.into() })
        })
        .collect()
}

/// `spd-say -L`: a `NAME LANGUAGE VARIANT` header, then one voice per line.
fn parse_dispatcher_voices(text: &str) -> Vec<Voice> {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("NAME"))
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next()?;
            let language = words.next().unwrap_or("");
            Some(Voice { id: name.into(), name: name.into(), language: language.into() })
        })
        .collect()
}

/// Standard base64 with padding.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let symbol = |v: u32| char::from(ALPHABET.get((v & 63) as usize).copied().unwrap_or(b'A'));
    let mut out = String::with_capacity(bytes.len().div_ceil(3).saturating_mul(4));
    for group in bytes.chunks(3) {
        let b = |i: usize| u32::from(group.get(i).copied().unwrap_or(0));
        let n = (b(0) << 16) | (b(1) << 8) | b(2);
        out.push(symbol(n >> 18));
        out.push(symbol(n >> 12));
        out.push(if group.len() > 1 { symbol(n >> 6) } else { '=' });
        out.push(if group.len() > 2 { symbol(n) } else { '=' });
    }
    out
}

/// `text` on one line: control characters and runs of white space become one space, and the
/// soft hyphen and zero-width characters go.
fn one_line(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if matches!(c, '\u{ad}' | '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}' | '\u{feff}') {
            continue;
        }
        if c.is_whitespace() || c.is_control() {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Cut page text into what to speak, in order: one sentence per chunk, at most
/// [`MAX_CHUNK_CHARS`] characters. Lines are joined (a word hyphenated across a line break is
/// rejoined), blank lines and form feeds end a paragraph, and chunks with no letters or digits
/// are dropped.
pub fn speech_chunks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for paragraph in paragraphs(text) {
        sentences(&paragraph, &mut out);
    }
    out
}

fn paragraphs(text: &str) -> Vec<String> {
    let mut list = Vec::new();
    // A form feed separates pages, so it ends a paragraph as a blank line does.
    for page in text.split('\x0c') {
        let mut current = String::new();
        for raw in page.split('\n') {
            let line = one_line(raw);
            if line.is_empty() {
                if !current.is_empty() {
                    list.push(std::mem::take(&mut current));
                }
            } else if current.is_empty() {
                current = line;
            } else if rejoins(&current, &line) {
                current.pop();
                current.push_str(&line);
            } else {
                current.push(' ');
                current.push_str(&line);
            }
        }
        if !current.is_empty() {
            list.push(current);
        }
    }
    list
}

/// Whether `line` continues a word `before` hyphenated at its end ("exam-" + "ple").
fn rejoins(before: &str, line: &str) -> bool {
    let mut end = before.chars().rev();
    end.next() == Some('-') && end.next().is_some_and(char::is_alphabetic) && line.chars().next().is_some_and(char::is_lowercase)
}

fn ends_sentence(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？' | '؟' | '।')
}

/// Full-width stops end a sentence with no space after them.
fn ends_without_space(c: char) -> bool {
    matches!(c, '。' | '！' | '？')
}

fn closes(c: char) -> bool {
    matches!(c, '"' | '\'' | '”' | '’' | ')' | ']' | '}' | '»' | '」' | '』')
}

fn sentences(paragraph: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = paragraph.chars().collect();
    let mut sentence = String::new();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        sentence.push(c);
        i += 1;
        if ends_sentence(c) {
            while let Some(&k) = chars.get(i).filter(|k| ends_sentence(**k) || closes(**k)) {
                sentence.push(k);
                i += 1;
            }
            if chars.get(i).is_none_or(|k| k.is_whitespace()) || ends_without_space(c) {
                push_sentence(std::mem::take(&mut sentence), out);
            }
        }
    }
    push_sentence(sentence, out);
}

/// Add `sentence`, cut between words into pieces of at most [`MAX_CHUNK_CHARS`] characters (a
/// longer word is cut between characters).
fn push_sentence(sentence: String, out: &mut Vec<String>) {
    let sentence = sentence.trim();
    if !sentence.chars().any(char::is_alphanumeric) {
        return;
    }
    let mut piece = String::new();
    let mut piece_chars = 0usize;
    for word in sentence.split(' ') {
        let mut word_chars = word.chars().count();
        if piece_chars > 0 && piece_chars + 1 + word_chars > MAX_CHUNK_CHARS {
            out.push(std::mem::take(&mut piece));
            piece_chars = 0;
        }
        if piece_chars > 0 {
            piece.push(' ');
            piece_chars += 1;
        }
        let mut chars = word.chars();
        while word_chars > MAX_CHUNK_CHARS.saturating_sub(piece_chars) {
            let room = MAX_CHUNK_CHARS.saturating_sub(piece_chars);
            piece.extend(chars.by_ref().take(room));
            out.push(std::mem::take(&mut piece));
            piece_chars = 0;
            word_chars -= room;
        }
        piece.extend(chars);
        piece_chars += word_chars;
    }
    if piece.chars().any(char::is_alphanumeric) {
        out.push(piece);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_standard_vectors() {
        for (input, want) in
            [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")]
        {
            assert_eq!(base64(input.as_bytes()), want);
        }
        assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
    }

    #[test]
    fn chunks_are_sentences_with_lines_joined() {
        let text = "It was a bright cold day in April, and the clocks were strik-\ning thirteen. Winston\nSmith hurried in!\n\nA new paragraph";
        assert_eq!(
            speech_chunks(text),
            ["It was a bright cold day in April, and the clocks were striking thirteen.", "Winston Smith hurried in!", "A new paragraph"]
        );
    }

    #[test]
    fn closing_quotes_stay_with_their_sentence() {
        assert_eq!(speech_chunks("He said \"stop.\" Then she left."), ["He said \"stop.\"", "Then she left."]);
        // A stop inside a number or a name doesn't end the sentence.
        assert_eq!(speech_chunks("Pi is 3.14 and e.g.x is odd."), ["Pi is 3.14 and e.g.x is odd."]);
    }

    #[test]
    fn full_width_stops_end_a_sentence_without_a_space() {
        assert_eq!(speech_chunks("今日は晴れ。明日は雨？"), ["今日は晴れ。", "明日は雨？"]);
    }

    #[test]
    fn form_feeds_and_blank_lines_end_paragraphs() {
        assert_eq!(speech_chunks("end of page\x0cnext page"), ["end of page", "next page"]);
        assert_eq!(speech_chunks("one\n \t\ntwo"), ["one", "two"]);
    }

    #[test]
    fn a_hyphen_before_a_capital_or_a_space_is_kept() {
        assert_eq!(speech_chunks("north-\nEast"), ["north- East"]);
        assert_eq!(speech_chunks("pages 3 -\nnine"), ["pages 3 - nine"]);
    }

    #[test]
    fn chunks_without_words_are_dropped() {
        assert!(speech_chunks("").is_empty());
        assert!(speech_chunks("\n\n — • … \x0c ").is_empty());
        assert_eq!(speech_chunks("•\n12"), ["• 12"]);
    }

    #[test]
    fn control_and_invisible_characters_are_removed() {
        assert_eq!(speech_chunks("soft\u{ad}ware\u{200b} and\u{0007}\u{0}bell"), ["software and bell"]);
    }

    #[test]
    fn long_sentences_are_cut_between_words_and_long_words_between_characters() {
        let words = vec!["word"; 300].join(" ");
        let chunks = speech_chunks(&words);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.chars().count() <= MAX_CHUNK_CHARS && !c.starts_with(' ') && !c.ends_with(' ')));
        assert_eq!(chunks.join(" "), words);

        let long = "é".repeat(MAX_CHUNK_CHARS * 2 + 7);
        let chunks = speech_chunks(&format!("a {long} b"));
        assert!(chunks.iter().all(|c| c.chars().count() <= MAX_CHUNK_CHARS));
        assert_eq!(chunks.concat().replace(' ', ""), format!("a{long}b"));
    }

    #[test]
    fn rates_map_to_each_engines_scale() {
        assert_eq!(sapi_rate(1.0), 0);
        assert_eq!(sapi_rate(3.0), 10);
        assert_eq!(sapi_rate(MIN_RATE), -6);
        assert_eq!(words_per_minute(1.0), 175);
        assert_eq!(words_per_minute(2.0), 350);
        assert_eq!(dispatcher_rate(1.5), 50);
        assert_eq!(dispatcher_volume(100), 100);
        assert_eq!(dispatcher_volume(0), -100);
        for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -4.0, 0.0, 99.0] {
            let r = SpeechSettings { rate, ..Default::default() }.rate();
            assert!((MIN_RATE..=MAX_RATE).contains(&r), "{rate} → {r}");
        }
        assert_eq!(SpeechSettings { rate: f32::NAN, ..Default::default() }.rate(), 1.0);
    }

    #[test]
    fn voice_names_that_look_like_options_are_refused() {
        let with = |v: &str| SpeechSettings { voice: Some(v.into()), ..Default::default() };
        assert!(matches!(with("-o/tmp/x").voice(), Err(SpeechError::InvalidVoice)));
        assert!(matches!(with("Zira\nrm").voice(), Err(SpeechError::InvalidVoice)));
        assert!(matches!(with(&"x".repeat(MAX_VOICE_CHARS + 1)).voice(), Err(SpeechError::InvalidVoice)));
        assert_eq!(with("  ").voice().ok(), Some(None));
        assert_eq!(with(" Microsoft Zira Desktop ").voice().ok(), Some(Some("Microsoft Zira Desktop")));
        let engine = Engine { kind: EngineKind::Espeak, program: PathBuf::from("espeak-ng") };
        assert!(matches!(Speaker::new(&engine, &with("--help")), Err(SpeechError::InvalidVoice)));
    }

    #[test]
    fn voice_lists_are_parsed() {
        let say = "Alex                en_US    # Most people recognize me by my voice.\nBad News            en_US    # The light you see\n";
        assert_eq!(
            parse_say_voices(say).iter().map(|v| (v.id.as_str(), v.language.as_str())).collect::<Vec<_>>(),
            [("Alex", "en_US"), ("Bad News", "en_US")]
        );
        let espeak = "Pty Language       Age/Gender VoiceName          File                 Other Languages\n 5  af              --/M      Afrikaans          gmw/af\n 5  en-gb           --/M      English_(Great_Britain) gmw/en\n";
        let v = parse_espeak_voices(espeak);
        assert_eq!(v.len(), 2);
        assert_eq!((v[1].id.as_str(), v[1].name.as_str()), ("en-gb", "English (Great Britain) (en-gb)"));
        let spd = "NAME                 LANGUAGE        VARIANT\nafrikaans            af              none\n";
        assert_eq!(parse_dispatcher_voices(spd), [Voice { id: "afrikaans".into(), name: "afrikaans".into(), language: "af".into() }]);
        let win = "Microsoft David Desktop\ten-US\r\nMicrosoft Zira Desktop\ten-US\r\n\r\n";
        assert_eq!(
            parse_windows_voices(win).iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            ["Microsoft David Desktop", "Microsoft Zira Desktop"]
        );
        // Junk never panics.
        for junk in ["", "\n\n", "#", "   #  x", "\t", "Pty\n 5"] {
            let _ = (parse_say_voices(junk), parse_espeak_voices(junk), parse_dispatcher_voices(junk), parse_windows_voices(junk));
        }
    }

    #[test]
    fn document_text_never_reaches_a_command_line_except_as_one_argument() {
        let text = "$(rm -rf /); `calc`; & del *";
        let settings = SpeechSettings::default();
        for kind in [EngineKind::Windows, EngineKind::Say, EngineKind::Espeak] {
            let command = speak_command(&Engine { kind, program: PathBuf::from("x") }, &settings, None, text).map_err(|e| e.to_string());
            let args: Vec<String> = command.map(|c| c.get_args().map(|a| a.to_string_lossy().into_owned()).collect()).unwrap_or_default();
            assert!(!args.iter().any(|a| a.contains("rm -rf") || a.contains("calc")), "{kind:?}: {args:?}");
        }
        let command = speak_command(&Engine { kind: EngineKind::SpeechDispatcher, program: PathBuf::from("x") }, &settings, None, text);
        let args: Vec<String> = command.map(|c| c.get_args().map(|a| a.to_string_lossy().into_owned()).collect()).unwrap_or_default();
        assert_eq!(args.iter().rev().take(2).collect::<Vec<_>>(), [text, "--"]);
    }

    #[test]
    fn a_stopper_stopped_first_refuses_to_speak() {
        let engine = Engine { kind: EngineKind::Espeak, program: PathBuf::from("pdfcraft-no-such-program") };
        let mut speaker = Speaker::new(&engine, &SpeechSettings::default()).unwrap();
        speaker.stopper().stop();
        assert!(matches!(speaker.speak("hello"), Err(SpeechError::Stopped)));
        // Nothing to say starts nothing.
        let mut speaker = Speaker::new(&engine, &SpeechSettings::default()).unwrap();
        assert!(speaker.speak(" \n\t").is_ok());
        assert!(matches!(speaker.speak("hello"), Err(SpeechError::Start(_))));
    }

    #[test]
    fn speech_dispatcher_cannot_record() {
        let engine = Engine { kind: EngineKind::SpeechDispatcher, program: PathBuf::from("x") };
        let out = std::env::temp_dir().join("pdfcraft-never-written.wav");
        assert!(matches!(record(&engine, &SpeechSettings::default(), &["hi".into()], &out, &Stopper::new()), Err(SpeechError::CannotRecord)));
    }

    /// Real speech: records a short WAV with this computer's engine. Opt in with
    /// `PDFCRAFT_SPEECH_TEST=1`; CI machines have no voices.
    #[test]
    fn records_a_wav_with_the_system_engine() {
        if std::env::var_os("PDFCRAFT_SPEECH_TEST").is_none() {
            return;
        }
        let Some(engine) = Engine::detect() else { return };
        let voices = voices(&engine).unwrap();
        assert!(!voices.is_empty(), "{} lists no voices", engine.name());
        let dir = std::env::temp_dir().join(format!("pdfcraft-speech-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("hello.wav");
        let chunks = speech_chunks("Hello from PdfCraft. Ünïcödé text — reads too!");
        record(&engine, &SpeechSettings { rate: 1.5, ..Default::default() }, &chunks, &out, &Stopper::new()).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert!(bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE"), "not a WAV file");
        assert!(bytes.len() > 10_000, "only {} bytes", bytes.len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
