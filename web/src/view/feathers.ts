/**
 * The look of the 3D hummingbird: a port of the "iridescent" shader of the
 * project's earlier Panda3D viewer (nv_affichage_Python_v2/hummingbird_viewer.py).
 *
 * The model's colour texture is very dark (its mean is about 15% of white), and
 * a physically based material shows it as such: flat, muddy feathers. The
 * earlier viewer did something simpler that looks much more like a
 * hummingbird, and we keep its choices and constants:
 *   - the colour texture is used as stored, without sRGB decoding, and scaled by
 *     an exposure of 4.2 (which also compensates the darkening of the occlusion map);
 *   - diffuse light from a warm key, a cool fill and an ambient term, the occlusion
 *     map darkening only the indirect part;
 *   - an iridescent accent: a sharp highlight and a Fresnel rim, both tinted by
 *     the model's specular-glossiness map (its colours are the blue and green
 *     glints of the feathers, its alpha the glossiness);
 *   - a soft clip that divides by the largest channel when it exceeds 1, which
 *     keeps the hue instead of washing it out;
 *   - feathers cut out where the colour texture's alpha is below 0.5, as in the
 *     original model (alpha mask).
 * Two things change. The key light is our sun, so the bird's shading agrees
 * with its shadow. And the ambient term is a quarter lower: the earlier viewer
 * showed the bird on a dark grey background, ours on a light sky, against which
 * the same shading looked flat.
 *
 * Two additions. In the sandbox, the other birds are tinted with their colour
 * (`tint`), so that each can be told apart. And a damaged wing turns orange-red in proportion to its damage
 * (fully at 40%), as on the drawn bird. Each vertex carries how much it belongs
 * to the left and to the right wing (`wingWeight`, from its skinning weights,
 * computed in bird.ts), so the colour fades smoothly into the body.
 */

import * as THREE from "three";

const vertexShader = /* glsl */ `
#include <common>
#include <skinning_pars_vertex>

attribute vec2 wingWeight;

varying vec2 vUv;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec2 vWing;

void main() {
  vUv = uv;
  vWing = wingWeight;
  #include <beginnormal_vertex>
  #include <skinbase_vertex>
  #include <skinnormal_vertex>
  #include <begin_vertex>
  #include <skinning_vertex>
  vec4 world = modelMatrix * vec4(transformed, 1.0);
  vWorldPos = world.xyz;
  vWorldNormal = normalize(mat3(modelMatrix) * objectNormal);
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const fragmentShader = /* glsl */ `
uniform sampler2D albedoTex;
uniform sampler2D specglossTex;
uniform sampler2D normalTex;
uniform sampler2D aoTex;
uniform vec3 keyDir;
uniform vec3 fillDir;
uniform float exposure;
uniform float alphaCutoff;
uniform vec2 damage;   // force lost by the (left, right) wing, 0 to 1
uniform vec4 tint;     // colour (raw albedo units) and amount: the sandbox's other birds are tinted

varying vec2 vUv;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec2 vWing;

const vec3 KEY_COL  = vec3(1.15, 1.10, 1.00);   // warm
const vec3 FILL_COL = vec3(0.38, 0.44, 0.55);   // cool and soft
const vec3 AMBIENT  = vec3(0.47, 0.48, 0.53);   // 75% of the earlier viewer's, for more contrast
const vec3 HURT     = vec3(0.19, 0.05, 0.012);  // orange-red, in the raw albedo's units (x exposure)

// Normal map without tangents: the surface frame is rebuilt from screen-space
// derivatives of the position and of the texture coordinates.
vec3 perturbNormal(vec3 pos, vec3 n, vec2 uv, vec3 mapN) {
  vec3 q0 = dFdx(pos);
  vec3 q1 = dFdy(pos);
  vec2 st0 = dFdx(uv);
  vec2 st1 = dFdy(uv);
  vec3 q1perp = cross(q1, n);
  vec3 q0perp = cross(n, q0);
  vec3 T = q1perp * st0.x + q0perp * st1.x;
  vec3 B = q1perp * st0.y + q0perp * st1.y;
  float det = max(dot(T, T), dot(B, B));
  float scale = det == 0.0 ? 0.0 : inversesqrt(det);
  return normalize(T * (mapN.x * scale) + B * (mapN.y * scale) + n * mapN.z);
}

void main() {
  vec4 albedo = texture2D(albedoTex, vUv);
  if (albedo.a < alphaCutoff) discard;
  albedo.rgb = mix(albedo.rgb, tint.rgb, tint.a);
  float hurt = clamp(dot(vWing, damage) / 0.4, 0.0, 1.0);
  albedo.rgb = mix(albedo.rgb, HURT, 0.8 * hurt);
  vec4 sg = texture2D(specglossTex, vUv);
  vec3 specColor = sg.rgb;   // the colour of the glints
  float gloss = sg.a;

  vec3 N = normalize(vWorldNormal);
  if (!gl_FrontFacing) N = -N;   // the wings are seen from both sides
  N = perturbNormal(vWorldPos, N, vUv, texture2D(normalTex, vUv).xyz * 2.0 - 1.0);
  vec3 V = normalize(cameraPosition - vWorldPos);
  float ndl = max(dot(N, keyDir), 0.0);
  float ndf = max(dot(N, fillDir), 0.0);

  // 1) diffuse: the colour texture under soft light; the occlusion darkens the indirect part
  float ao = texture2D(aoTex, vUv).r;
  vec3 diffuse = albedo.rgb * (AMBIENT * ao + KEY_COL * ndl + FILL_COL * ndf * ao);

  // 2) iridescent accent: sharp highlight and Fresnel rim, tinted by the specular map
  vec3 H = normalize(V + keyDir);
  float shininess = mix(16.0, 400.0, gloss);
  float spec = pow(max(dot(N, H), 0.0), shininess) * ndl;
  float fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0);
  vec3 irid = specColor * (spec * 1.2 + fresnel * 0.28);

  vec3 color = (diffuse + irid) * exposure;
  color /= max(max(color.r, color.g), max(color.b, 1.0));   // soft clip that keeps the hue
  gl_FragColor = vec4(color, 1.0);
}
`;

/** Textures read as stored (no sRGB decoding): the shader's constants were tuned that way. */
function raw(texture: THREE.Texture | null): THREE.Texture | null {
  if (texture) {
    texture.colorSpace = THREE.NoColorSpace;
    texture.needsUpdate = true;
  }
  return texture;
}

/**
 * Replaces the material of one mesh of the model. `original` is the material the
 * glTF loader made (we take its textures); `specGloss` the model's specular-glossiness map.
 */
export function featherMaterial(
  original: THREE.MeshStandardMaterial,
  specGloss: THREE.Texture,
  keyDir: THREE.Vector3,
): THREE.ShaderMaterial {
  const white = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
  white.needsUpdate = true;
  const flat = new THREE.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1);
  flat.needsUpdate = true;
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
    uniforms: {
      albedoTex: { value: raw(original.map) ?? white },
      specglossTex: { value: raw(specGloss) },
      normalTex: { value: raw(original.normalMap) ?? flat },
      aoTex: { value: raw(original.aoMap) ?? white },
      keyDir: { value: keyDir.clone().normalize() },
      fillDir: { value: new THREE.Vector3(-0.6, 0.35, -0.7).normalize() },
      exposure: { value: 4.2 },
      alphaCutoff: { value: 0.5 },
      damage: { value: new THREE.Vector2(0, 0) },
      tint: { value: new THREE.Vector4(0, 0, 0, 0) },
    },
  });
}

/** The shadow must use the same cut-out feathers as the colour. */
export function featherShadowMaterial(original: THREE.MeshStandardMaterial): THREE.MeshDepthMaterial {
  return new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: original.map, alphaTest: 0.5 });
}
