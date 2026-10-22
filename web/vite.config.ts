import { defineConfig } from "vitest/config";

export default defineConfig({
  // Relative paths, so that the built site also works from a sub-folder
  // such as https://<user>.github.io/hummingbird-rl/ (GitHub Pages).
  base: "./",
  // three.js alone is about 550 kB once minified (140 kB compressed); that is expected.
  build: { chunkSizeWarningLimit: 800 },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
