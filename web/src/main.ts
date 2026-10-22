/**
 * Setup check: a three.js scene with the TypeScript simulator running live.
 *
 * Not the viewer yet (that comes with the bird model and the Watch mode).
 * It shows that Node, Vite, TypeScript and three.js work, and that the
 * simulator port runs in the browser: the bird, a small sphere with a cone
 * for its beak, is given the hover action with a small push and no feedback.
 * Hovering is unstable, so it drifts and swings more and more, and hits the
 * ground after about 5 s; then the flight restarts. The controllers fix that.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { hoverAction } from "./sim/aero";
import { type BirdState, BirdDynamics, checkLimits, stateFromEuler } from "./sim/dynamics";
import { DEFAULT_PARAMS, hoverAmplitude } from "./sim/params";
import { START } from "./sim/world";

const p = DEFAULT_PARAMS;

// --- scene -----------------------------------------------------------------------------
const canvas = document.querySelector<HTMLCanvasElement>("#view")!;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe9f3);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a9a7a, 2.0));

// Same axes as the simulator: x forward, y up, z to the right; units in metres.
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
camera.position.set(0.35, 0.62, 0.45); // about 60 cm from the start point
const controls = new OrbitControls(camera, canvas);
controls.target.set(START[0], START[1], START[2]);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(1.5, 64), // the arena: 1.5 m around the centre
  new THREE.MeshLambertMaterial({ color: 0xb7c9a8 }),
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);
const grid = new THREE.GridHelper(3, 30, 0x6b7f5e, 0x9fb294); // 10 cm squares
grid.position.y = 0.001;
scene.add(grid);

const bird = new THREE.Group();
const body = new THREE.Mesh(new THREE.SphereGeometry(p.r_bird, 24, 16), new THREE.MeshLambertMaterial({ color: 0x2e8b57 }));
const beak = new THREE.Mesh(new THREE.ConeGeometry(0.006, 0.03, 12), new THREE.MeshLambertMaterial({ color: 0x333333 }));
beak.rotation.z = -Math.PI / 2; // the cone points along +y by default; turn it to +x, the beak axis
beak.position.x = p.r_bird + 0.012;
bird.add(body, beak);
scene.add(bird);

const marker = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8), new THREE.MeshBasicMaterial({ color: 0xd9480f }));
marker.position.set(START[0], START[1], START[2]);
scene.add(marker);

// --- simulation --------------------------------------------------------------------------
const dyn = new BirdDynamics(p);
const action = hoverAction(p);
let state: BirdState = newFlight();
let restartAt: number | null = null;

function newFlight(): BirdState {
  return stateFromEuler(START, [0.05, 0.0, 0.0]); // a 5 cm/s push forward
}

const status = document.querySelector<HTMLDivElement>("#status")!;
const intro =
  `<b>Web setup OK</b> · three.js r${THREE.REVISION} · simulator loaded ` +
  `(hover amplitude ${((hoverAmplitude(p) * 180) / Math.PI).toFixed(1)}°).<br>` +
  `The bird gets the hover action and a 5 cm/s push, with no controller. Drag to orbit, scroll to zoom.`;

// Fixed physics step with a time accumulator: each frame advances the simulation
// by the real time elapsed, in steps of 1/240 s (at most 64 per frame).
let last = performance.now();
let accumulator = 0.0;

function frame(now: number): void {
  accumulator += Math.min((now - last) / 1000, 0.25);
  last = now;
  let steps = 0;
  while (accumulator >= p.dt && steps < 64 && restartAt === null) {
    [state] = dyn.step(state, action);
    accumulator -= p.dt;
    steps++;
    const reason = checkLimits(state);
    if (reason !== null) restartAt = now + 1000;
    status.innerHTML = `${intro}<br>t = ${state.time.toFixed(2)} s${reason ? ` · <b>${reason.replaceAll("_", " ")}</b>` : ""}`;
  }
  if (restartAt !== null && now > restartAt) {
    state = newFlight();
    restartAt = null;
    accumulator = 0.0;
  }

  bird.position.set(state.pos[0], state.pos[1], state.pos[2]);
  const [w, x, y, z] = state.quat;
  bird.quaternion.set(x, y, z, w); // three.js stores (x, y, z, w)

  const { clientWidth: width, clientHeight: height } = canvas;
  if (canvas.width !== Math.floor(width * renderer.getPixelRatio()) || canvas.height !== Math.floor(height * renderer.getPixelRatio())) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
