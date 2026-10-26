HEALTH COACH VOICE PROTOTYPE - PORTABLE
=======================================

Runs on Windows 10/11 (64-bit). Nothing to install, no internet needed.

TO RUN
1. Copy this whole folder to the laptop's Desktop (much faster than running
   from the USB drive, because Gemma loads 3+ GB from disk).
2. Double-click "Start Demo.bat".
   - If Windows shows "Windows protected your PC", click "More info", then
     "Run anyway". This happens because the program is not code-signed.
3. Wait until the black window says "Ready" (about 1 minute the first time).
   The browser opens on its own. If it doesn't, open Chrome or Edge and go to
   the address shown in the window (usually http://127.0.0.1:8000).
4. Allow the microphone when the browser asks.
5. Tap the mic once and talk. With Hands-free on, the mic reopens after each reply.

TO STOP
Close the black "Health Coach Demo" window. That stops everything.
If anything seems stuck, double-click "Stop Demo.bat".

BEFORE THE MEETING
- Do one full dry run on this laptop: start it, say a sentence, ask a plan
  question. The first Gemma answer after starting is the slowest.
- Check the laptop's microphone and speakers (a headset is better in a room).
- Plug the laptop in: speech and language models run faster on mains power.

IF SOMETHING GOES WRONG
- A red "Speech in", "Gemma" or "Voice" chip at the top of the page: hover it
  for the reason. ollama.log in this folder has Gemma's startup messages.
- Microphone does nothing: use the address exactly as shown (127.0.0.1 or
  localhost), click the lock/mic icon in the address bar, and allow it.
- To start fresh, close the window, delete demo.db in this folder, start again.

The folder also works if copied anywhere else; it writes only demo.db and
ollama.log next to itself.
