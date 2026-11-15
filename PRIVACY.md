# Privacy - Hupesi TTS

**Hupesi TTS does not collect your data.** No account is required, there are
no ads, and no usage statistics or error reports are sent to the author.

## Where your data is stored

Everything you enter and create stays only on your machine:

- Text, generated audio, and subtitles (History tab);
- Cloned voices and rights declarations (confirmation content, timestamp, mic/file
  source, and terms version), stored together with the voice data files;
- Pronunciation dictionary and settings;
- Terms-acceptance marker (stores only the accepted terms version number).

The data folder locations are described in the README, in the "Where your data is
stored" section. Uninstalling the app or deleting those folders removes all of it.

## When the app uses the network

After installation, the app can run completely without Internet access. It connects
to the outside only in the cases below, and only to download; it does not send
your content:

| Case | Connects to | Default |
|---|---|---|
| Check for updates (enabled in Settings, or by clicking "Check for updates") | GitHub, at most once per day | **Off** |
| You click Download for a model not bundled with the app | Hugging Face | Only when you click |
| You click Download for the Kokoro model or add Japanese/Chinese (phonemizer install) | PyPI, GitHub / releases.astral.sh | Only when you click |
| You click "Install GPU support" | PyPI, download.pytorch.org, GitHub / releases.astral.sh | Only when you click |
| You click a link inside the app | That website, opened in your browser | Only when you click |

As with any Internet connection, these servers can see your IP address and the
software name (`CIT-Voice-Studio/<version>`). The app does not include your text,
audio, or any personal information.

## Recording a microphone sample or importing audio to create a voice

When recording from a microphone, you read any content naturally for about 3–8
seconds as a voice sample. When importing an audio file, the app decodes and
keeps a sample of at most 15 seconds in the interface, then sends this WAV sample
to a service running on your own machine to create the voice. There is no voice
cloning service on the author's servers. Both microphone and audio file imports
use the same rights declaration; the app does not perform speech recognition on
the sample.

The original audio file is not automatically copied into the library. The
temporary WAV sample is deleted after processing; the reusable voice data and the
rights declaration remain on your machine. Deleting a voice in the app also
deletes its associated declaration. The original file you selected and any
backups you made yourself are not deleted. Voices saved from older versions are
not automatically given a new declaration.

The app does not collect legal names, identity documents, or account information,
and it does not send voices, declarations, or recognition text to the author. The
declaration is a user confirmation, not identity verification.

## Allowing other machines to access the API

This feature is **off by default**. When you enable it, other machines on your
network can call your machine's speech API using an API key. Text they send is
processed on your machine. Connections within a local network use unencrypted
HTTP, so enable this only on networks you trust.

## Generated audio

WAV/MP3 files exported by the app include the note "AI-generated speech … CIT
Voice Studio" in the file metadata (to label AI-generated content). This note
does not contain any personal information.