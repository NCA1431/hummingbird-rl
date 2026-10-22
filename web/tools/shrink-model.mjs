// Shrinks the hummingbird model for the web (24.7 MB -> about 1.1 MB): drops the unused
// "flight" animation and the redundant keyframes, resizes the textures to 1024 px and
// re-encodes them as WebP. The model is "Animated Hovering Flying Hummingbird Loop" by
// LasquetiSpice (Sketchfab, CC BY 4.0); these changes are listed in the file's extras,
// as the licence asks.
//
// The model also contains a specular-glossiness map that its (repaired) material no longer
// uses; the viewer's shader needs it (see src/view/feathers.ts), so we save it as a
// separate WebP file before prune() drops it.
//
// Run once, from web/tools (the tools are not part of the site's dependencies):
//   npm install --no-save @gltf-transform/core@4.5.1 @gltf-transform/functions@4.5.1 @gltf-transform/extensions@4.5.1 sharp@0.35.5
//   node shrink-model.mjs <original .glb> ../public/models/hummingbird.glb ../public/models/hummingbird_specgloss.webp

import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { dedup, prune, resample, textureCompress } from "@gltf-transform/functions";
import sharp from "sharp";

const [input, output, specOutput] = process.argv.slice(2);
if (!input || !output || !specOutput) {
  throw new Error("usage: node shrink-model.mjs <in.glb> <out.glb> <specgloss.webp>");
}

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(input);

// The texture no material refers to is the specular-glossiness map (RGB: colour of the
// glints, A: glossiness).
const unused = doc.getRoot().listTextures().filter((t) => !t.listParents().some((p) => p.propertyType === "Material"));
if (unused.length !== 1) throw new Error(`expected one unused texture, found ${unused.length}`);
await sharp(Buffer.from(unused[0].getImage())).resize(1024, 1024).webp({ quality: 85, alphaQuality: 90 }).toFile(specOutput);

for (const anim of doc.getRoot().listAnimations()) {
  if (anim.getName() === "hover") continue;
  // Its samplers hold the keyframe data: dispose of them too, or prune() keeps it.
  for (const c of anim.listChannels()) c.dispose();
  for (const s of anim.listSamplers()) s.dispose();
  anim.dispose();
}
await doc.transform(
  resample(), // drops keyframes that a straight interpolation would reproduce (bones that do not move)
  prune(),
  dedup(),
  textureCompress({ encoder: sharp, targetFormat: "webp", resize: [1024, 1024], quality: 82 }),
);
doc.getRoot().getAsset().extras = {
  source:
    "Animated Hovering Flying Hummingbird Loop by LasquetiSpice, https://sketchfab.com/3d-models/animated-hovering-flying-hummingbird-loop-102884713a2742ce829e368d2a790c45",
  license: "CC BY 4.0, https://creativecommons.org/licenses/by/4.0/",
  changes: "flight animation removed, redundant keyframes removed, textures resized to 1024 px and converted to WebP",
};
await io.write(output, doc);
