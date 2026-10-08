import { NeuralNet, makeDataset } from './neural.mjs';

const $ = id => document.getElementById(id);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const fmt = n => (n >= 0 ? '+' : '') + n.toFixed(3);
let net, samples, running = false, seen = 0, batches = 0, history = [], dataset = 'xor', lastPlot = 0;
const lowMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
const plotCanvas = document.createElement('canvas');
plotCanvas.width = plotCanvas.height = 82;
const plotContext = plotCanvas.getContext('2d', { willReadFrequently: true });
let pointsPerStep = 32;

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1,Math.round(rect.width * dpr));
  const h = Math.max(1,Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { ctx: canvas.getContext('2d'), w, h, dpr };
}
function settingValues() {
  return { hiddenLayers: +$('layers').value, width: +$('neurons').value, activation: $('activation').value,
    optimizer: $('optimizer').value, learningRate: +$('lr').value, seed: 42 };
}
function syncLabels() {
  $('layersValue').textContent = $('layers').value;
  $('neuronsValue').textContent = $('neurons').value;
  $('lrValue').textContent = (+$('lr').value).toFixed(3);
  $('speedValue').textContent = $('speed').value + ' steps/frame';
  $('probeXValue').textContent = (+$('probeX').value >= 0 ? '+' : '') + (+$('probeX').value).toFixed(2);
  $('probeYValue').textContent = (+$('probeY').value >= 0 ? '+' : '') + (+$('probeY').value).toFixed(2);
}
function setRunning(value) {
  running = value;
  $('trainText').textContent = running ? 'PAUSE TRAINING' : 'START TRAINING';
  $('trainIcon').textContent = running ? 'Ⅱ' : '▶';
  $('trainStatus').textContent = running ? 'LEARNING · BACKPROPAGATION IN PROGRESS' : batches ? 'PAUSED · REAL WEIGHTS PRESERVED' : 'READY · REAL BACKPROPAGATION';
}
function createExperiment() {
  setRunning(false); net = new NeuralNet(settingValues()); samples = makeDataset(dataset, 1024);
  seen = 0; batches = 0; history = [];
  $('historyEnd').textContent = 'CURRENT';
  syncLabels();
  refresh(true);
}
function stepOnce() {
  const batch = [];
  // Sequential cyclic batches through a shuffled dataset (every sample is covered).
  for (let i = 0; i < pointsPerStep; i++) batch.push(samples[(seen + i) % samples.length]);
  net.trainBatch(batch);
  seen += pointsPerStep; batches++;
}
function trainingFrame() {
  if (running) {
    const n = +$('speed').value;
    for (let i = 0; i < n; i++) stepOnce();
    if (performance.now() - lastPlot > 130) { refresh(false); lastPlot = performance.now(); }
  }
  requestAnimationFrame(trainingFrame);
}
function refresh(first = false) {
  const m = net.evaluate(samples);
  $('accuracy').textContent = (100*m.accuracy).toFixed(1) + '%';
  $('loss').textContent = m.loss.toFixed(4);
  $('epochs').textContent = Math.floor(seen/samples.length).toLocaleString();
  if (first || batches && (history.length === 0 || batches !== history.at(-1).batch)) {
    history.push({ batch: batches, epoch: seen / samples.length, loss: m.loss, acc: m.accuracy });
    if (history.length > 350) history = history.filter((_,i) => i % 2 === 0);
  }
  $('historyEnd').textContent = Math.floor(seen/samples.length) + ' EPOCHS';
  drawDecision(); drawHistory(); drawTopology(); showProbe();
}
function drawDecision() {
  const {ctx,w,h} = fitCanvas($('decision'));
  const N = 82, img = plotContext.createImageData(N,N);
  for (let py = 0; py < N; py++) for (let px = 0; px < N; px++) {
    const p = net.predict([px/(N-1)*2-1, 1-py/(N-1)*2]);
    const ix = (py*N+px)*4;
    // Warm coral class 0 → luminous cyan class 1. The actual prediction chooses the blend.
    img.data[ix] = Math.round(153*(1-p)+29*p);
    img.data[ix+1] = Math.round(65*(1-p)+129*p);
    img.data[ix+2] = Math.round(102*(1-p)+172*p);
    img.data[ix+3] = 255;
  }
  plotContext.putImageData(img,0,0);
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0,0,w,h);
  ctx.drawImage(plotCanvas,0,0,w,h);
  ctx.strokeStyle='rgba(255,255,255,.12)'; ctx.lineWidth=Math.max(1,w/700);
  for(let i=1;i<4;i++){let x=i*w/4,y=i*h/4;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  ctx.strokeStyle='rgba(255,255,255,.28)'; ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  const radius=clamp(w/175,2.5,5);
  for(const sample of samples) {
    const x=(sample.x[0]+1)*.5*w, y=(1-sample.x[1])*.5*h;
    ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2);
    ctx.fillStyle=sample.y?'#68f3f0':'#ff9e83';ctx.fill();
    ctx.strokeStyle='rgba(9,20,37,.84)';ctx.lineWidth=Math.max(1.4,w/380);ctx.stroke();
  }
}
function drawHistory() {
  const {ctx,w,h} = fitCanvas($('history'));
  ctx.clearRect(0,0,w,h);
  const pad={l:33,r:8,t:7,b:20}, uw=w-pad.l-pad.r,uh=h-pad.t-pad.b;
  const maxY=Math.max(.7, ...history.map(v=>v.loss))*1.12;
  ctx.font=Math.max(10,w/42)+'px monospace';
  for(let i=0;i<=3;i++){
    const y=pad.t+uh*i/3;
    ctx.strokeStyle='rgba(135,164,201,.13)';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillStyle='#6985a5';ctx.textAlign='right';ctx.fillText((maxY*(1-i/3)).toFixed(1),pad.l-8,y+4);
  }
  if(!history.length) return;
  const xof=i=>pad.l+uw*(history.length===1?0:i/(history.length-1));
  const yof=v=>pad.t+uh*(1-clamp(v.loss/maxY,0,1));
  const grad=ctx.createLinearGradient(0,pad.t,0,h);grad.addColorStop(0,'rgba(87,231,233,.3)');grad.addColorStop(1,'rgba(87,231,233,0)');
  ctx.beginPath();ctx.moveTo(xof(0),h-pad.b);
  history.forEach((v,i)=>ctx.lineTo(xof(i),yof(v)));
  ctx.lineTo(xof(history.length-1),h-pad.b);ctx.closePath();ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();history.forEach((v,i)=>i?ctx.lineTo(xof(i),yof(v)):ctx.moveTo(xof(i),yof(v)));
  ctx.strokeStyle='#71eff0';ctx.lineWidth=Math.max(2,w/190);ctx.shadowColor='#43dfe8';ctx.shadowBlur=10;ctx.stroke();ctx.shadowBlur=0;
  const y=yof(history.at(-1));ctx.beginPath();ctx.arc(xof(history.length-1),y,Math.max(3,w/150),0,Math.PI*2);ctx.fillStyle='#b7ffff';ctx.fill();
}
function drawTopology() {
  const {ctx,w,h} = fitCanvas($('topology'));
  ctx.clearRect(0,0,w,h);
  const sizes=net.sizes, pos=[];
  const padX=Math.min(46,w*.1),padY=Math.min(24,h*.13);
  for(let l=0;l<sizes.length;l++){
    const x=padX+(w-2*padX)*l/(sizes.length-1);
    pos[l]=Array.from({length:sizes[l]},(_,j)=>({x,y:padY+(h-2*padY)*(j+1)/(sizes[l]+1)}));
  }
  for(let l=0;l<net.weights.length;l++)for(let j=0;j<net.weights[l].length;j++)for(let k=0;k<net.weights[l][j].length;k++){
    const weight=net.weights[l][j][k],a=pos[l][k],b=pos[l+1][j];
    ctx.strokeStyle=weight>=0?'rgba(94,233,238,'+clamp(.12+Math.abs(weight)*.24,.12,.85)+')':'rgba(255,150,120,'+clamp(.12+Math.abs(weight)*.24,.12,.8)+')';
    ctx.lineWidth=clamp(Math.abs(weight)*1.4, .7, 3)*Math.min(w/400,1.5);
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  }
  for(let l=0;l<sizes.length;l++)for(let j=0;j<sizes[l];j++){
    const {x,y}=pos[l][j],r=clamp(w/115,3.2,6.5),glow=ctx.createRadialGradient(x-r/3,y-r/3,0,x,y,r*1.3);
    glow.addColorStop(0,'#d9ffff');glow.addColorStop(.35,l===0?'#95bfff':l===sizes.length-1?'#a2e7f3':'#7bece5');glow.addColorStop(1,'#255480');
    ctx.fillStyle=glow;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=l===1&&j===0?'#fff':'rgba(185,240,255,.36)';ctx.lineWidth=l===1&&j===0?2.2:1;ctx.stroke();
  }
  ctx.fillStyle='#7594af';ctx.font=Math.max(9,w/46)+'px sans-serif';ctx.textAlign='center';
  pos.forEach((col,i)=>ctx.fillText(i===0?'INPUT':i===pos.length-1?'OUTPUT':'HIDDEN '+i,col[0].x,h-3));
}
function showProbe() {
  if (!net) return;
  const x=[+$('probeX').value,+$('probeY').value], result=net.forward(x);
  $('weight1').textContent=fmt(net.weights[0][0][0]);
  $('weight2').textContent=fmt(net.weights[0][0][1]);
  $('bias').textContent=fmt(net.biases[0][0]);
  $('weightedSum').textContent=fmt(result.Z[0][0]);
  $('hiddenOutput').textContent=fmt(result.A[1][0]);
  $('probePrediction').textContent=(result.probability*100).toFixed(2)+'%';
}
for(const id of ['layers','neurons','activation','optimizer','lr']) $(id).addEventListener('change',createExperiment);
for(const id of ['layers','neurons','lr','speed','probeX','probeY']) $(id).addEventListener('input',()=>{syncLabels();if(id==='probeX'||id==='probeY')showProbe();});
for(const button of document.querySelectorAll('[data-set]')) button.addEventListener('click',()=>{
  dataset=button.dataset.set;
  for(const b of document.querySelectorAll('[data-set]')){b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',String(b===button));}
  createExperiment();
});
$('train').addEventListener('click',()=>setRunning(!running));
$('step').addEventListener('click',()=>{setRunning(false);stepOnce();refresh();});
$('reset').addEventListener('click',createExperiment);
window.addEventListener('resize',()=>{if(net)refresh(true);});
for(const link of document.querySelectorAll('.sidebar nav a'))link.addEventListener('click',()=>{
  document.querySelectorAll('.sidebar nav a').forEach(el=>el.classList.remove('active'));link.classList.add('active');
});

createExperiment();
requestAnimationFrame(trainingFrame);

// WebGL is progressive enhancement. The CSS glass neuron remains if WebGL or the CDN is unavailable.
async function start3D() {
  const canvas=$('neuron3d'), sceneTag=document.querySelector('.scene-tag');
  try {
    if(!window.WebGLRenderingContext) throw new Error('WebGL unavailable');
    const THREE=await import('https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js');
    const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true,powerPreference:'low-power'});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.8));
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(43,1,.1,100);
    camera.position.set(0,0,10.8);
    scene.add(new THREE.AmbientLight(0x9bc5ff,2.1));
    const key=new THREE.PointLight(0x9dfaff,55,16);key.position.set(-3,4,5);scene.add(key);
    const rim=new THREE.PointLight(0x7478ff,65,18);rim.position.set(3,-2,-2);scene.add(rim);
    const group=new THREE.Group();scene.add(group);
    const glass=new THREE.MeshPhysicalMaterial({color:0x5caee5,metalness:.04,roughness:.06,transmission:.89,thickness:1.5,ior:1.35,transparent:true,opacity:.98,clearcoat:1,clearcoatRoughness:.05,side:THREE.DoubleSide});
    const orb=new THREE.Mesh(new THREE.SphereGeometry(1.17,64,48),glass);group.add(orb);
    const inner=new THREE.Mesh(new THREE.SphereGeometry(.74,48,32),new THREE.MeshStandardMaterial({color:0x2155a2,emissive:0x1761b4,emissiveIntensity:.65,roughness:.2,metalness:.33,transparent:true,opacity:.75}));group.add(inner);
    const aura=new THREE.Mesh(new THREE.SphereGeometry(1.30,48,32),new THREE.MeshBasicMaterial({color:0x3d9fff,transparent:true,opacity:.10,side:THREE.BackSide,depthWrite:false,blending:THREE.AdditiveBlending}));group.add(aura);
    const ringmat=new THREE.MeshBasicMaterial({color:0x8dedff,transparent:true,opacity:.44,blending:THREE.AdditiveBlending,depthWrite:false});
    for(const rot of [[.3,.15,.1],[1.13,-.35,.8],[1.4,.8,-.4]]){
      const ring=new THREE.Mesh(new THREE.TorusGeometry(1.43,.008,5,120),ringmat.clone());
      ring.rotation.set(...rot);group.add(ring);
    }
    function labelTexture(text) {
      const c=document.createElement('canvas');c.width=512;c.height=256;
      const ct=c.getContext('2d');
      ct.clearRect(0,0,512,256);ct.textAlign='center';ct.textBaseline='middle';
      ct.shadowColor='#79fcff';ct.shadowBlur=24;ct.fillStyle='#f3ffff';ct.font='bold 130px sans-serif';ct.fillText(text,256,128);
      const tex=new THREE.CanvasTexture(c);tex.colorSpace=THREE.SRGBColorSpace;return tex;
    }
    const symbol=new THREE.Sprite(new THREE.SpriteMaterial({map:labelTexture('Σ  ƒ'),transparent:true,depthTest:false}));
    symbol.scale.set(1.45,.72,1);symbol.position.set(0,0,1.32);group.add(symbol);
    const curveMat=new THREE.MeshBasicMaterial({color:0x4cc4ff,transparent:true,opacity:.34,blending:THREE.AdditiveBlending,depthWrite:false});
    const pulseMat=new THREE.MeshBasicMaterial({color:0x9cfbff,transparent:true,blending:THREE.AdditiveBlending});
    const satelliteMat=new THREE.MeshPhysicalMaterial({color:0x7bdaed,roughness:.12,metalness:.1,transmission:.35,thickness:.5,clearcoat:1});
    const coordinates=[[-3.6,1.7,-.5],[-3.9,0,0],[-3.6,-1.7,.5],[3.55,.85,-.2],[3.55,-.9,.4]];
    const links=[];
    coordinates.forEach((v,i)=>{
      const sat=new THREE.Mesh(new THREE.SphereGeometry(i<3?.24:.3,30,24),satelliteMat);
      sat.position.set(...v);group.add(sat);
      const left=i<3;
      const start=new THREE.Vector3(...v);
      const end=new THREE.Vector3(left?-.95:1.0,left?(v[1]*.20):(v[1]*.35),.1);
      const mid=new THREE.Vector3((start.x+end.x)*.5,v[1]*.72,v[2]*.3+.3);
      const path=new THREE.QuadraticBezierCurve3(start,mid,end);
      const tube=new THREE.Mesh(new THREE.TubeGeometry(path,48,.019,7,false),curveMat);
      group.add(tube);
      const particle=new THREE.Mesh(new THREE.SphereGeometry(.058,10,10),pulseMat);group.add(particle);
      links.push({path,particle,offset:i*.17,left});
    });
    const dotGeom=new THREE.SphereGeometry(.014,6,5);
    const dotMat=new THREE.MeshBasicMaterial({color:0x70cdff,transparent:true,opacity:.7});
    const dots=new THREE.InstancedMesh(dotGeom,dotMat,95),dummy=new THREE.Object3D(),rand=(s=>()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;})(87);
    for(let i=0;i<95;i++){
      const theta=rand()*Math.PI*2,r=1.9+rand()*2.7;
      dummy.position.set(Math.cos(theta)*r,(rand()-.5)*4.0,(rand()-.5)*2.7);
      dummy.scale.setScalar(.5+rand()*1.9);dummy.updateMatrix();dots.setMatrixAt(i,dummy.matrix);
    }
    group.add(dots);
    let targetX=-.12,targetY=.08,dragging=false,lastX=0,lastY=0,zoom=10.8,rotationTime=0;
    canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{if(!dragging)return;targetY+=(e.clientX-lastX)*.006;targetX+=(e.clientY-lastY)*.005;targetX=clamp(targetX,-.8,.8);lastX=e.clientX;lastY=e.clientY;});
    canvas.addEventListener('pointerup',()=>dragging=false);
    canvas.addEventListener('pointercancel',()=>dragging=false);
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=clamp(zoom+e.deltaY*.012,7.5,15);},{passive:false});
    const resize=()=>{
      const w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight);
      renderer.setSize(w,h,false);
      camera.aspect=w/h;camera.fov=w<480?57:43;camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(canvas);resize();
    document.querySelector('.scene').classList.add('webgl-ready');
    sceneTag.textContent='WEBGL · LIVE SIGNAL';
    function animate(now){
      if(!document.body.contains(canvas)){renderer.dispose();return;}
      const time=now*.001;rotationTime=time;
      group.rotation.y+=(targetY-group.rotation.y)*.035;
      group.rotation.x+=(targetX-group.rotation.x)*.035;
      if(!dragging&&!lowMotion)targetY+=.0005;
      camera.position.z+=(zoom-camera.position.z)*.1;
      orb.rotation.y=time*.08;inner.rotation.y=-time*.08;
      for(const link of links){
        const t=lowMotion?.55:(time*.21+(link.offset||0))%1;
        link.particle.position.copy(link.path.getPoint(link.left?t:1-t));
        link.particle.scale.setScalar(.7+.5*Math.sin(time*5));
      }
      renderer.render(scene,camera);
      if(!lowMotion)requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  } catch(error) {
    sceneTag.textContent='GLASS · 2D FALLBACK';
    canvas.style.pointerEvents='none';
    console.info('3D enhancement unavailable; using built-in neuron illustration.',error?.message||error);
  }
}
start3D();
