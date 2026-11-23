// =====================================================================
// NovaFit 3D avataras (three.js). Naudoja assets/avatar3d/body.glb ir body.json,
// kuriuos sukuria tools/mh_export_glb.py iš MakeHuman (CC0) duomenų.
//
//   const av = await createAvatar3D(canvas);
//   av.setBody({ gender:"female", height:170, weight:65, waist:72, hips:96, arm:60 });
//   av.setOutfit([{ type:"tshirt", color:"#ffffff" }, ...]);
//
// Kas tikra: figūra keičiasi pagal lytį, ūgį, svorį, juosmenį, klubus ir rankų ilgį (MakeHuman formos).
// Kas apytikslė: centimetrai → formos stiprumas skaičiuojamas pagal modelio matavimus,
// drabužiai – kūno paviršius, atitrauktas nuo odos (laisvi drabužiai ir sijonas supaprastinti).
// =====================================================================
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BASE = new URL("./avatar3d/", import.meta.url);
const LEVEL_BMI = [17.5, 20, 22.5, 26.5, 31];     // tie patys kaip 2D figūrų lygiai w0…w4
// Kalibravimas: kiek „tikrų“ centimetrų atitinka modelio matavimas (numatytieji programėlės skaičiai = 0 pokytis)
const CAL = { waist: 72 / 65.3, hips: 96 / 89.8, arm: 60 / 44.3 };

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// KMI → MakeHuman svoris 0…1 (tolygiai tarp lygių)
function weightFromBmi(bmi) {
  if (bmi <= LEVEL_BMI[0]) return 0;
  for (let i = 0; i < LEVEL_BMI.length - 1; i++)
    if (bmi <= LEVEL_BMI[i + 1]) return (i + (bmi - LEVEL_BMI[i]) / (LEVEL_BMI[i + 1] - LEVEL_BMI[i])) / 4;
  return 1;
}

export async function createAvatar3D(canvas, opts = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: !!opts.keepBuffer });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.05, 30);
  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = false; controls.enableDamping = true;
  controls.minDistance = 2.2; controls.maxDistance = 7;
  controls.minPolarAngle = 0.9; controls.maxPolarAngle = 1.75;
  scene.add(new THREE.HemisphereLight("#ffffff", "#bdb7cf", 1.5));
  const key = new THREE.DirectionalLight("#ffffff", 1.7); key.position.set(-1.5, 2.6, 2.4); scene.add(key);
  const rim = new THREE.DirectionalLight("#ffffff", 0.6); rim.position.set(1.8, 1.5, -2); scene.add(rim);

  // Šešėlis po kojomis
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.33, 48),
    new THREE.MeshBasicMaterial({ color: "#16151B", transparent: true, opacity: 0.1, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.002; shadow.scale.set(1, 0.45, 1);
  scene.add(shadow);

  const [gltf, info] = await Promise.all([
    new GLTFLoader().loadAsync(new URL("body.glb", BASE).href),
    fetch(new URL("body.json", BASE)).then(r => r.json())
  ]);
  const body = gltf.scene.getObjectByName("avatar");
  const geo = body.geometry;
  const dict = body.morphTargetDictionary;
  const root = new THREE.Group(); scene.add(root);
  root.add(body);
  // Sijono / palto apačios „varpai“ – atskiri modeliai su tomis pačiomis formomis
  const bells = {};
  for (const type of Object.keys(info.garments)) {
    const m = gltf.scene.getObjectByName("bell-" + type);
    if (!m) continue;
    m.material = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.85, metalness: 0, side: THREE.DoubleSide });
    m.morphTargetInfluences = body.morphTargetInfluences;
    m.visible = false; m.frustumCulled = false;
    root.add(m);
    bells[type] = m;
  }

  // --- Oda + juodi apatiniai (tolygus kraštas pagal kūno lauką) ---
  const shared = { uMale: { value: 0 } };
  body.material = new THREE.MeshStandardMaterial({ color: "#E2D6C8", roughness: 0.75, metalness: 0 });
  body.material.onBeforeCompile = sh => {
    sh.uniforms.uMale = shared.uMale;
    sh.uniforms.uUnder = { value: new THREE.Color("#1A1A1D") };
    sh.vertexShader = "attribute float _underf;\nattribute float _underm;\nuniform float uMale;\nvarying float vUnder;\n" +
      sh.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\n vUnder = mix(_underf, _underm, uMale);");
    sh.fragmentShader = "varying float vUnder;\nuniform vec3 uUnder;\n" +
      sh.fragmentShader.replace("#include <color_fragment>",
        "#include <color_fragment>\n { float w = fwidth(vUnder); diffuseColor.rgb = mix(diffuseColor.rgb, uUnder, smoothstep(-w, w, vUnder)); }");
  };

  // --- Drabužiai: ta pati geometrija, atitraukta nuo odos ir apkirpta ---
  const garments = {};
  function garment(type) {
    if (garments[type]) return garments[type];
    const g = info.garments[type];
    if (!g) return null;
    const attr = "_g_" + type;                       // GLTFLoader pavadinimus paverčia mažosiomis raidėmis
    const u = { uThick: { value: g.thick }, uFlare: { value: g.flare }, uFlareY: { value: g.flareY }, uSmooth: { value: g.smooth ? 1 : 0 } };
    const mat = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.85, metalness: 0, side: THREE.DoubleSide });
    mat.onBeforeCompile = sh => {
      Object.assign(sh.uniforms, u);
      sh.vertexShader = `attribute float ${attr};\nattribute vec3 _smooth;\nuniform float uThick, uFlare, uFlareY, uSmooth;\nvarying float vF;\n` +
        sh.vertexShader.replace("#include <morphtarget_vertex>",
          `#include <morphtarget_vertex>
           transformed += _smooth * uSmooth;
           transformed += normalize(objectNormal) * (uThick + uFlare * max(0.0, uFlareY - position.y));
           vF = ${attr};`);
      sh.fragmentShader = "varying float vF;\n" +
        sh.fragmentShader.replace("void main() {", "void main() {\n if (vF < 0.0) discard;");
    };
    mat.customProgramCacheKey = () => "garment-" + type;
    const mesh = new THREE.Mesh(geo, mat);
    mesh.morphTargetInfluences = body.morphTargetInfluences;   // tos pačios formos kaip kūnui
    mesh.morphTargetDictionary = dict;
    mesh.visible = false; mesh.frustumCulled = false;
    root.add(mesh);
    return garments[type] = mesh;
  }

  // --- Atskiri Blenderyje sukurti drabužiai (kraunami tik pirmą kartą užsivilkus) ---
  // Jų formos turi tuos pačius pavadinimus ir eiliškumą kaip kūno, todėl keičiasi kartu su figūra.
  const MODELS = { hoodie: "hoodie.glb", jeans: "jeans.glb", sweater: "sweater.glb" };
  const models = {}, loading = {};
  let lastItems = [];
  function loadModel(type) {
    if (!loading[type]) loading[type] = new GLTFLoader().loadAsync(new URL(MODELS[type], BASE).href).then(g => {
      let m = null; g.scene.traverse(o => { if (o.isMesh && !m) m = o; });
      const names = (m.geometry.userData.targetNames) || Object.keys(m.morphTargetDictionary || {});
      m.material = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.9, metalness: 0, side: THREE.DoubleSide });
      m.morphTargetInfluences = body.morphTargetInfluences;      // tas pats formų masyvas kaip kūno
      m.morphTargetDictionary = dict;
      m.frustumCulled = false; m.visible = false;
      m.position.set(0, 0, 0); m.rotation.set(0, 0, 0); m.scale.set(1, 1, 1);
      root.add(m);
      models[type] = m;
      setOutfit(lastItems);                                      // pritaikom dabartinį derinį
      return m;
    }).catch(err => { console.error("Nepavyko įkelti", type, err); });
    return loading[type];
  }

  // --- Kūno formos pagal matmenis ---
  let natural = info.measures["female-avg"].height;
  function measureAt(gender, w, key, morph) {
    // modelio matavimas tarp plono / vidutinio / stambaus (tiesiškai)
    const pick = lvl => info.measures[`${gender}-${lvl}${morph ? "|" + morph : ""}`][key];
    return w < 0.5 ? lerp(pick("thin"), pick("avg"), w / 0.5) : lerp(pick("avg"), pick("heavy"), (w - 0.5) / 0.5);
  }
  function setBody(p) {
    const inf = body.morphTargetInfluences;
    inf.fill(0);
    const male = p.gender === "male", g = male ? "male" : "female", s = male ? "m" : "f";
    const w = weightFromBmi(p.weight / ((p.height / 100) ** 2));
    inf[dict.male] = male ? 1 : 0;
    if (w < 0.5) inf[dict[`${s}-thin`]] = (0.5 - w) / 0.5; else inf[dict[`${s}-heavy`]] = (w - 0.5) / 0.5;
    shared.uMale.value = male ? 1 : 0;
    natural = info.measures[`${g}-avg`].height;
    const k = p.height / natural;                       // ūgis – visos figūros mastelis
    root.scale.setScalar(k);
    // Juosmuo, klubai, rankos: skirtumas nuo tipiško (tokio pat ūgio ir svorio) → formos stiprumas
    const shape = (key, entered, plus, minus, cal) => {
      if (!entered) return;
      const typical = measureAt(g, w, key) * cal * k;
      const diff = entered - typical;
      const per = (diff >= 0 ? measureAt(g, w, key, plus) - measureAt(g, w, key) : measureAt(g, w, key) - measureAt(g, w, key, minus)) * cal * k;
      const t = per > 0.1 ? clamp(Math.abs(diff) / per, 0, 1.6) : 0;
      return { name: diff >= 0 ? plus : minus, t };
    };
    for (const r of [shape("waist", p.waist, "waist+", "waist-", CAL.waist), shape("hips", p.hips, "hips+", "hips-", CAL.hips)])
      if (r) inf[dict[r.name]] = r.t;
    if (p.arm) {   // rankos ilgis = žastas + dilbis, pokytis padalinamas per pusę
      const typical = measureAt(g, w, "arm") * CAL.arm * k, diff = p.arm - typical;
      for (const part of ["upperarm", "lowerarm"]) {
        const name = part + (diff >= 0 ? "+" : "-");
        const per = Math.abs(measureAt(g, w, "arm", name) - measureAt(g, w, "arm")) * CAL.arm * k;
        if (per > 0.1) inf[dict[name]] = clamp(Math.abs(diff) / 2 / per, 0, 1.6);
      }
    }
    frame(k);
  }

  // --- Derinys ---
  function setOutfit(items) {
    lastItems = items;
    for (const m of [...Object.values(garments), ...Object.values(bells), ...Object.values(models)]) m.visible = false;
    const outer = items.some(i => i.type === "jacket" || i.type === "coat");
    for (const it of items) {
      // Po striuke ar paltu laisvi marškinėliai „susispaudžia“ (kaip 2D versijoje)
      const type = it.type === "oversize" && outer ? "tshirt" : it.type;
      if (MODELS[type]) {
        if (models[type]) { models[type].material.color.set(it.color); models[type].visible = true; }
        else loadModel(type);
        continue;
      }
      const m = garment(type);
      if (!m) continue;
      m.material.color.set(it.color);
      m.visible = true;
      if (bells[type]) { bells[type].material.color.set(it.color); bells[type].visible = true; }
    }
  }

  // --- Kamera ir piešimas ---
  let lastK = 1;
  function frame(k = lastK) {
    lastK = k;
    const h = 1.6 * k * (natural / 159);
    controls.target.set(0, h * 0.52, 0);
    if (!frame.done) { camera.position.set(0, h * 0.58, h * 2.55); frame.done = true; }
    controls.update();
  }
  function resize() {
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 400;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();
  let running = true;
  renderer.setAnimationLoop(() => { if (!running) return; controls.update(); renderer.render(scene, camera); });

  return {
    setBody, setOutfit,
    resetView() { frame.done = false; frame(); },
    pause(v) { running = !v; },
    render() { controls.update(); renderer.render(scene, camera); },
    info
  };
}
