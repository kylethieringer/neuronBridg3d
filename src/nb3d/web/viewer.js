// 3D overlay viewer: EM cells (magenta), LM image (green), optional full expression (orange),
// template bounding box (grey). Coordinates are template microns; y is flipped so anterior is up.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLORS = { em: 0xd63fd6, lm: 0x33cc33, ex: 0xf0a020, box: 0x666670 };
const MIDLINE_X = 132.12;

export class Viewer {
  constructor(el) {
    this.el = el;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x0d0d10);
    el.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.layers = { em: {}, lm: null, ex: null, box: null };
    this.pointSize = 1.25;
    this.lmOpacity = 0.35;
    this.mirror = false;
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); };
    loop();
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _points(xyz, color, opacity, mirror = false) {
    const pos = new Float32Array(xyz.length * 3);
    xyz.forEach(([x, y, z], i) => {
      pos[i * 3] = mirror ? 2 * MIDLINE_X - x : x;
      pos[i * 3 + 1] = -y;
      pos[i * 3 + 2] = -z;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const m = new THREE.PointsMaterial({ color, size: this.pointSize, sizeAttenuation: true,
      transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
    return new THREE.Points(g, m);
  }

  _clear(obj) {
    if (!obj) return;
    this.root.remove(obj);
    obj.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
  }

  // One EM cell: its skeleton as lines plus its nodes as points, drawn on top of the LM cloud
  // (no depth test, rendered last) so the cell stays visible where the LM signal covers it.
  _emCell(xyz, segments, mirror) {
    const g = new THREE.Group();
    if (segments && segments.length) {
      const pos = new Float32Array(segments.length * 6);
      segments.forEach((s, i) => {
        for (let k = 0; k < 2; k++) {
          const x = s[k * 3];
          pos[i * 6 + k * 3] = mirror ? 2 * MIDLINE_X - x : x;
          pos[i * 6 + k * 3 + 1] = -s[k * 3 + 1];
          pos[i * 6 + k * 3 + 2] = -s[k * 3 + 2];
        }
      });
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color: COLORS.em, depthTest: false }));
      lines.renderOrder = 10;
      g.add(lines);
    }
    const pts = this._points(xyz, COLORS.em, 1, mirror);
    pts.material.depthTest = false;
    pts.renderOrder = 10;
    g.add(pts);
    g.userData.points = pts;
    return g;
  }

  _drawEm(visible = {}) {
    Object.values(this.layers.em).forEach(o => this._clear(o));
    this.layers.em = {};
    for (const [id, xyz] of Object.entries(this.emData)) {
      const c = this._emCell(xyz, this.emSegments?.[id], this.mirror);
      c.visible = visible[id] ?? true;
      this.layers.em[id] = c; this.root.add(c);
    }
  }

  setData({ em, em_segments, lm, template_outline, expression }) {
    this._clear(this.layers.lm); this._clear(this.layers.ex); this._clear(this.layers.box);
    this.emData = em;
    this.emSegments = em_segments;
    this._drawEm();
    this.layers.lm = this._points(lm, COLORS.lm, this.lmOpacity); this.root.add(this.layers.lm);
    if (expression) { this.layers.ex = this._points(expression, COLORS.ex, 0.25); this.root.add(this.layers.ex); }
    const xs = template_outline.map(p => p[0]), ys = template_outline.map(p => p[1]), zs = template_outline.map(p => p[2]);
    const size = [Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), Math.max(...zs) - Math.min(...zs)];
    const box = new THREE.Box3(new THREE.Vector3(0, -size[1], -size[2]), new THREE.Vector3(size[0], 0, 0));
    this.layers.box = new THREE.Box3Helper(box, COLORS.box); this.root.add(this.layers.box);
    if (!this.framed) this.resetView();
  }

  resetView() {
    const c = new THREE.Vector3(132, -258, -76);
    this.controls.target.copy(c);
    this.camera.position.set(c.x, c.y, c.z + 700);
    this.camera.up.set(0, 1, 0);
    this.framed = true;
  }

  setCellVisible(id, on) { if (this.layers.em[id]) this.layers.em[id].visible = on; }

  setPointSize(s) {
    this.pointSize = s;
    for (const o of [...Object.values(this.layers.em).map(c => c.userData.points), this.layers.lm, this.layers.ex]) if (o) o.material.size = s;
  }

  setOpacity(a) {
    this.lmOpacity = a;
    if (this.layers.lm) { this.layers.lm.material.opacity = a; this.layers.lm.material.transparent = a < 1; }
  }

  setMirror(on) {
    this.mirror = on;
    if (!this.emData) return;
    this._drawEm(Object.fromEntries(Object.entries(this.layers.em).map(([k, o]) => [k, o.visible])));
  }
}
