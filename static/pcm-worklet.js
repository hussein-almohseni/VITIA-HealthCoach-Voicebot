// Captures microphone audio, downsamples it to 16 kHz mono, and posts 100 ms chunks
// (1600 float32 samples) to the page, which forwards them to the server.
class PCMCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.chunk = new Float32Array(1600);
    this.n = 0;
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      // Averaging the samples in each step is a simple low-pass filter before decimation.
      this.sum += channel[i];
      this.count += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        this.chunk[this.n++] = this.sum / this.count;
        this.sum = 0;
        this.count = 0;
        if (this.n === this.chunk.length) {
          this.port.postMessage(this.chunk.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PCMCapture);
