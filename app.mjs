const $ = id => document.getElementById(id);
let report = null;
const urls = new Map();
$('transcript-file').addEventListener('change',async()=>{
  report=null;$('report').hidden=true;$('export').disabled=true;
  try {
    const file=$('transcript-file').files[0];if(!file)return;
    if(file.size>16000)throw new Error('Transcript exceeds 16 KB');
    const text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());
    if(text.length>4000)throw new Error('Transcript exceeds 4000 characters');
    $('transcript').value=text;$('status').textContent='Transcript loaded';
  } catch(error){$('transcript').value='';$('status').textContent=error.message;}
});
for (const id of ['threshold','mfcc','transcript']) $(id).addEventListener('input',()=>{
  report=null; $('report').hidden=true; $('export').disabled=true;
  $('status').textContent='Analysis settings changed';
});
for (const name of ['baseline', 'practice']) {
  $(name).addEventListener('change', () => {
    if (urls.has(name)) URL.revokeObjectURL(urls.get(name));
    const file = $(name).files[0];
    if (file) { const url = URL.createObjectURL(file); urls.set(name,url); $(`${name}-player`).src = url; }
    else { urls.delete(name); $(`${name}-player`).removeAttribute('src'); }
    report = null; $('report').hidden = true; $('export').disabled = true;
    $('status').textContent = 'Recording selection changed';
  });
}
async function encoded(file) {
  if (file.size > 20 * 1024 * 1024) throw new Error('Each recording must be at most 20 MiB');
  return new Promise((resolve,reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = () => reject(new Error('Recording could not be read'));
    reader.readAsDataURL(file);
  });
}
async function run(action) {
  $('analyze').disabled = $('fixture').disabled = $('export').disabled = true;
  $('baseline').disabled = $('practice').disabled = $('threshold').disabled = $('mfcc').disabled = $('transcript').disabled = $('transcript-file').disabled = true;
  $('status').textContent = 'Analyzing';
  report = null; $('report').hidden = true;
  try {
    const response = await action();
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Analysis failed');
    report = result; render(); $('status').textContent = 'Analysis complete'; $('export').disabled = false;
  } catch(error) { $('status').textContent = error.message; }
  finally { $('analyze').disabled = $('fixture').disabled = false; $('baseline').disabled = $('practice').disabled = $('threshold').disabled = $('mfcc').disabled = $('transcript').disabled = $('transcript-file').disabled = false; }
}
$('inputs').addEventListener('submit', event => {
  event.preventDefault();
  const baseline = $('baseline').files[0], practice = $('practice').files[0];
  run(async () => fetch('/api/compare', {method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({baseline:await encoded(baseline),practice:await encoded(practice),thresholdDb:Number($('threshold').value),includeMfcc:$('mfcc').checked,transcript:$('transcript').value})}));
});
$('fixture').addEventListener('click', () => run(() => fetch('/api/fixture')));
function render() {
  $('report').hidden = false;
  $('duration').textContent = `${report.baseline.durationSeconds.toFixed(2)} seconds`;
  $('count').textContent = `${report.events.length} acoustic regions`;
  $('source').textContent = report.synthetic ? 'SYNTHETIC TONES - NOT SPEECH' : 'Uploaded recordings';
  $('rubric-score').textContent=`Acoustic fidelity ${report.rubric.score.toFixed(1)} / 100 (custom rubric)`;
  $('events').replaceChildren(); $('limits').replaceChildren();
  for (const event of report.events) {
    const row = document.createElement('tr');
    const kind = document.createElement('td'), time = document.createElement('td'), evidence = document.createElement('td');
    kind.textContent = event.kind.replaceAll('-', ' ');
    time.textContent = `${event.start.toFixed(2)} - ${event.end.toFixed(2)} s`;
    if (event.kind === 'additional-silence') evidence.textContent = 'Baseline frames above relative activity threshold; practice frames below it. Not a semantic judgment.';
    if (event.kind === 'relative-energy-change') {
      const average = event.evidence.reduce((sum,e) => sum+e.normalizedDeltaDb,0)/event.evidence.length;
      evidence.textContent = `${average.toFixed(2)} dB average normalized energy delta; threshold ${report.thresholdDb} dB.`;
    }
    if (event.kind === 'signal-clipping') evidence.textContent = 'At least 2% of samples reach digital full scale in each flagged frame.';
    row.append(kind,time,evidence); $('events').append(row);
  }
  if (!report.events.length) { const row=document.createElement('tr'), cell=document.createElement('td'); cell.colSpan=3; cell.textContent='No regions exceeded the configured thresholds'; row.append(cell); $('events').append(row); }
  $('spectral').hidden = !report.mfcc;
  if (report.mfcc) $('spectral-summary').textContent = `Mean C1-C12 distance: ${report.mfcc.meanDistance.toFixed(3)} (uncalibrated)`;
  $('alignment').hidden = !report.alignment;
  $('words').replaceChildren();
  if (report.alignment) {
    $('alignment-status').textContent = 'Model-derived word boundaries; review required. No time warp applied.';
    const base=report.alignment.baseline.words, attempt=report.alignment.practice.words;
    base.forEach((word,index)=>{
      const other=attempt[index],row=document.createElement('tr');
      for(const value of [word.word,`${word.start.toFixed(2)} - ${word.end.toFixed(2)} s`,other ? `${other.start.toFixed(2)} - ${other.end.toFixed(2)} s` : 'Missing',other ? `${(other.start-word.start).toFixed(2)} s` : 'Missing']) {
        const cell=document.createElement('td');cell.textContent=value;row.append(cell);
      }
      $('words').append(row);
    });
  }
  for (const text of [...report.limitations,report.rubric.interpretation,report.rubric.definition,...report.rubric.limitations,...(report.mfcc?.limitations || [])]) { const li=document.createElement('li'); li.textContent=text; $('limits').append(li); }
  if(report.alignment) for(const text of new Set([...report.alignment.baseline.warnings,...report.alignment.practice.warnings])) {const li=document.createElement('li');li.textContent=text;$('limits').append(li);}
  draw();
}
function draw() {
  if (!report) return;
  const canvas=$('timeline'), ratio=window.devicePixelRatio||1;
  const width=canvas.clientWidth, height=280;
  canvas.width=Math.round(width*ratio); canvas.height=height*ratio;
  const ctx=canvas.getContext('2d'); ctx.scale(ratio,ratio);
  const left=48, right=width-18, top=24, bottom=height-32;
  const duration=report.baseline.durationSeconds;
  const x=t=>left+t/duration*(right-left), y=db=>bottom-(Math.max(-40,Math.min(20,db))+40)/60*(bottom-top);
  ctx.font='11px system-ui';
  for (const db of [-40,-20,0,20]) { ctx.strokeStyle='#e0e7e3'; ctx.beginPath();ctx.moveTo(left,y(db));ctx.lineTo(right,y(db));ctx.stroke();ctx.fillStyle='#53625d';ctx.fillText(`${db} dB`,3,y(db)+4); }
  for (const event of report.events) {ctx.fillStyle='#f3deb84d';ctx.fillRect(x(event.start),top,x(event.end)-x(event.start),bottom-top);}
  for (const [key,color] of [['baseline','#08705b'],['practice','#b35282']]) {ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();report[key].frames.forEach((frame,i)=>{if(i)ctx.lineTo(x(frame.time),y(frame.relativeDb));else ctx.moveTo(x(frame.time),y(frame.relativeDb));});ctx.stroke();}
  ctx.fillStyle='#53625d';ctx.fillText('0 s',left,height-10);ctx.fillText(`${duration.toFixed(2)} s`,right-45,height-10);
  if (report.mfcc) drawMfcc();
}
function drawMfcc() {
  const canvas=$('mfcc-timeline'), ratio=window.devicePixelRatio||1, width=canvas.clientWidth, height=160;
  canvas.width=Math.round(width*ratio); canvas.height=height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  const peak=Math.max(1,...report.mfcc.frames.map(f=>f.distance));
  ctx.strokeStyle='#b35282';ctx.lineWidth=2;ctx.beginPath();
  report.mfcc.frames.forEach((frame,i)=>{const x=48+frame.time/report.baseline.durationSeconds*(width-66),y=136-frame.distance/peak*112;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
  ctx.fillStyle='#53625d';ctx.font='11px system-ui';ctx.fillText(peak.toFixed(2),3,24);ctx.fillText('0',24,136);
}
new ResizeObserver(draw).observe($('timeline'));
$('export').addEventListener('click',()=>{
  if(!report)return;
  const a=document.createElement('a');a.href=report.downloadUrl;a.download='cadencelab-report.json';document.body.append(a);a.click();a.remove();
});
