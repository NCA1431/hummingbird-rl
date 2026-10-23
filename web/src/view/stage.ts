/**
 * The 3D stage: renderer, camera, lights, the arena floor, and the animation loop.
 *
 * Units are metres and the axes are the simulator's (x forward, y up, z to the
 * right), which are also three.js's. The arena is the 1.5 m circle the bird
 * must not leave; the grid has 10 cm squares, a useful ruler for a 10 cm bird.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class Stage {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(45, 1, 0.005, 50);
  readonly controls: OrbitControls;
  readonly sunDirection = new THREE.Vector3(1.2, 3.0, 0.8).normalize(); // towards the sun
  private readonly sun: THREE.DirectionalLight;
  private last = performance.now();
  private inEye = false; // first-person view: the orbit controls are off

  constructor(private readonly canvas: HTMLCanvasElement) {
    // alpha: the sky is the page's CSS gradient behind the canvas.
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;

    this.scene.add(new THREE.HemisphereLight(0xf2f6ff, 0x7d8c6a, 1.6));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.8);
    this.sun.position.copy(this.sunDirection).multiplyScalar(3.4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const cam = this.sun.shadow.camera;
    cam.left = cam.bottom = -1.6;
    cam.right = cam.top = 1.6;
    cam.near = 0.5;
    cam.far = 6;
    this.scene.add(this.sun, this.sun.target);

    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(1.5, 96),
      new THREE.MeshStandardMaterial({ color: 0xa9bf96, roughness: 1 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    const grid = new THREE.GridHelper(3, 30, 0x6f845f, 0x93a883);
    grid.position.y = 0.0005;
    const edge = new THREE.Mesh(
      new THREE.RingGeometry(1.49, 1.5, 128),
      new THREE.MeshBasicMaterial({ color: 0x55663f, side: THREE.DoubleSide }),
    );
    edge.rotation.x = -Math.PI / 2;
    edge.position.y = 0.001;
    this.scene.add(ground, grid, edge);

    this.camera.position.set(0.35, 0.65, 0.45);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.5, 0);
    this.controls.minDistance = 0.05;
    this.controls.maxDistance = 6;
    this.controls.enableDamping = true;
  }

  /** Calls `frame(realSeconds)` before every rendered frame. */
  /** Called just before each render, once the camera is placed (the air's depth pass uses it). */
  readonly beforeRender: (() => void)[] = [];

  run(frame: (dt: number) => void): void {
    const loop = (now: number) => {
      const dt = Math.min((now - this.last) / 1000, 0.1);
      this.last = now;
      this.resize();
      frame(dt);
      if (!this.inEye) this.controls.update();
      for (const f of this.beforeRender) f();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  /** Moves the orbit centre to `p`, carrying the camera along (follow mode). */
  follow(p: THREE.Vector3): void {
    const delta = p.clone().sub(this.controls.target);
    this.controls.target.add(delta);
    this.camera.position.add(delta);
  }

  /** Puts the camera at `offset` from `target` and looks at it. */
  view(target: THREE.Vector3, offset: THREE.Vector3): void {
    this.leaveEye();
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(offset);
  }

  /**
   * First-person view from `pos`, looking along the heading `yaw` (rad) and slightly down. The
   * horizon stays level, as a hummingbird keeps its head level while its body rocks; the
   * body's own roll and pitch would make the view unwatchable. A wider lens than the orbit
   * camera's, closer to a bird's field of view (still far from its nearly 300°).
   */
  eye(pos: THREE.Vector3, yaw: number): void {
    if (!this.inEye) {
      this.inEye = true;
      this.controls.enabled = false;
      this.camera.fov = 75;
      this.camera.updateProjectionMatrix();
    }
    this.camera.position.copy(pos);
    this.camera.rotation.set(-0.12, yaw - Math.PI / 2, 0, "YXZ"); // the camera looks along -z, the bird along +x
  }

  private leaveEye(): void {
    if (!this.inEye) return;
    this.inEye = false;
    this.controls.enabled = true;
    this.camera.fov = 45;
    this.camera.updateProjectionMatrix();
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.canvas;
    const ratio = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.floor(w * ratio) || this.canvas.height !== Math.floor(h * ratio)) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }
}
