"""Build self-contained HTML widgets (no CDN) from widget_data.json.  Usage: python build_widgets.py out_dir"""
import json, sys
OUT = sys.argv[1]
D = json.load(open(f"{OUT}/widget_data.json"))
CSS = """<style>
.wbox{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:1.2em 0;background:#fafbfc;font-size:.95em}
.wbox svg,.wbox canvas{display:block;max-width:100%;height:auto}
.wbox .wrow{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin-top:8px}
.wbox label{cursor:pointer;white-space:nowrap}
.wbox button{font:inherit;padding:2px 10px;border:1px solid #aab;border-radius:5px;background:#fff;cursor:pointer}
.wbox .wn{font-variant-numeric:tabular-nums;font-weight:700}
.wbox .cap{color:#555;font-size:.9em;margin-top:6px}
</style>"""
HELP = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}"""
B, G, R = "#0072B2", "#00795A", "#C84B00"

def wrap(name, body):
    html = "```{=html}\n" + CSS + "\n" + body + "\n```\n"
    open(f"{OUT}/{name}.html", "w", encoding="utf8").write(html)

# ---------------- I3 ----------------
body3 = ("""<div class="wbox" id="w-i3">
<svg viewBox="0 0 560 300" style="touch-action:none;max-width:560px"></svg>
<div class="wrow"><span>Slope w = <span class="wn" id="i3w"></span> mpg/hp</span><span>Intercept b = <span class="wn" id="i3b"></span> mpg</span><button id="i3r">Reset</button></div>
<div class="cap">Drag the two end points of the green line to place it where you think it fits the three cars best. No score is shown.</div></div>
<script>(function(){
""" + HELP + """
const D=__D__;const svg=document.querySelector('#w-i3 svg');
const X0=120,X1=175,Y0=10,Y1=24,L=50,Rr=540,T=15,Bt=255;
const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T),iy=py=>Y0+(Bt-py)/(Bt-T)*(Y1-Y0);
const HX=[122,173],init=[21,13.5];let hy=init.slice(),drag=-1;
function draw(){svg.innerHTML='';
 for(let x=120;x<=170;x+=10){el(svg,'line',{x1:sx(x),x2:sx(x),y1:Bt,y2:Bt+4,stroke:'#888'});el(svg,'text',{x:sx(x),y:Bt+17,'text-anchor':'middle','font-size':12,fill:'#444'},x);}
 for(let y=10;y<=24;y+=2){el(svg,'line',{x1:L-4,x2:L,y1:sy(y),y2:sy(y),stroke:'#888'});el(svg,'text',{x:L-8,y:sy(y)+4,'text-anchor':'end','font-size':12,fill:'#444'},y);}
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 el(svg,'text',{x:(L+Rr)/2,y:290,'text-anchor':'middle','font-size':13,fill:'#333'},'Horsepower (hp)');
 el(svg,'text',{x:14,y:(T+Bt)/2,'text-anchor':'middle','font-size':13,fill:'#333',transform:'rotate(-90 14 '+(T+Bt)/2+')'},'mpg');
 el(svg,'line',{x1:sx(HX[0]),y1:sy(hy[0]),x2:sx(HX[1]),y2:sy(hy[1]),stroke:'""" + G + """','stroke-width':3});
 D.hp.forEach((x,i)=>el(svg,'circle',{cx:sx(x),cy:sy(D.mpg[i]),r:7,fill:'""" + B + """'}));
 HX.forEach((x,i)=>{const c=el(svg,'circle',{cx:sx(x),cy:sy(hy[i]),r:10,fill:'#fff',stroke:'""" + G + """','stroke-width':3,style:'cursor:ns-resize'});
  c.addEventListener('pointerdown',e=>{drag=i;c.setPointerCapture(e.pointerId);});});
 const w=(hy[1]-hy[0])/(HX[1]-HX[0]),b=hy[0]-w*HX[0];
 document.getElementById('i3w').textContent=w.toFixed(3);document.getElementById('i3b').textContent=b.toFixed(1);}
svg.addEventListener('pointermove',e=>{if(drag<0)return;const r=svg.getBoundingClientRect();const py=(e.clientY-r.top)*300/r.height;hy[drag]=Math.max(Y0,Math.min(Y1,iy(py)));draw();});
svg.addEventListener('pointerup',()=>{drag=-1;});
document.getElementById('i3r').onclick=()=>{hy=init.slice();draw();};draw();})();</script>""").replace("__D__", json.dumps(D["I3"]))
wrap("I3", body3)

# ---------------- I6 ----------------
names = ["Cylinders", "Displacement", "Horsepower (hp)", "Weight (lb)", "Acceleration", "Model year"]
boxes = "".join(f'<label><input type="checkbox" data-i="{i}"{" checked" if i==2 else ""}> {n}</label>' for i, n in enumerate(names))
wrap("I6", f"""<div class="wbox" id="w-i6">
<div class="wrow">{boxes}</div>
<svg viewBox="0 0 560 90" style="max-width:560px"></svg>
<div class="wrow"><span>Coefficient of hp = <span class="wn" id="i6c">-</span> mpg/hp</span><span>Training RMSE = <span class="wn" id="i6r">-</span> mpg</span></div>
<div class="cap">Tick or untick features and watch how the coefficient of hp changes. The data are the 313 training cars.</div></div>
<script>(function(){{
{HELP}
const D=__D__;const root=document.getElementById('w-i6'),svg=root.querySelector('svg');
const L=30,Rr=530,lo=-0.2,hi=0.05,sx=v=>L+(v-lo)/(hi-lo)*(Rr-L);
function upd(){{let m=0;root.querySelectorAll('input').forEach(c=>{{if(c.checked)m|=1<<(+c.dataset.i);}});
 svg.innerHTML='';const s=D.sets[m];
 for(let v=-0.2;v<=0.051;v+=0.05){{el(svg,'text',{{x:sx(v),y:80,'text-anchor':'middle','font-size':12,fill:'#444'}},(Math.abs(v)<1e-9?'0':v.toFixed(2)));el(svg,'line',{{x1:sx(v),x2:sx(v),y1:50,y2:55,stroke:'#888'}});}}
 el(svg,'line',{{x1:L,x2:Rr,y1:52,y2:52,stroke:'#888'}});el(svg,'line',{{x1:sx(0),x2:sx(0),y1:15,y2:55,stroke:'#555','stroke-dasharray':'4 3'}});
 const c=document.getElementById('i6c'),r=document.getElementById('i6r');
 if(!s){{c.textContent='-';r.textContent='-';el(svg,'text',{{x:280,y:35,'text-anchor':'middle','font-size':13,fill:'#555'}},'Select at least one feature');return;}}
 r.textContent=s.rmse.toFixed(2);
 if(s.hp===null){{c.textContent='hp not selected';el(svg,'text',{{x:280,y:35,'text-anchor':'middle','font-size':13,fill:'#555'}},'hp is not in the model');}}
 else{{c.textContent=s.hp.toFixed(3);el(svg,'line',{{x1:sx(0),x2:sx(s.hp),y1:34,y2:34,stroke:'{G}','stroke-width':10}});el(svg,'circle',{{cx:sx(s.hp),cy:34,r:8,fill:'{G}'}});}}
}}
root.querySelectorAll('input').forEach(c=>c.addEventListener('change',upd));upd();}})();</script>""".replace("__D__", json.dumps(D["I6"])))

# ---------------- I7 ----------------
wrap("I7", f"""<div class="wbox" id="w-i7">
<svg viewBox="0 0 560 300" style="max-width:560px"></svg>
<div class="wrow"><label>Training cars used to fit <input type="range" id="i7s" min="0" max="1000" value="250" style="width:240px"></label>
<span>n = <span class="wn" id="i7n"></span></span><span>Training RMSE <span class="wn" id="i7a"></span></span><span>Test RMSE <span class="wn" id="i7b"></span></span></div>
<div class="cap">Six features, 7 parameters. The test set is always the same 79 cars. Curves come from one fixed random ordering of the training cars. The horizontal axis is logarithmic and the vertical axis is cut off above 9.</div></div>
<script>(function(){{
{HELP}
const D=__D__;const svg=document.querySelector('#w-i7 svg'),sl=document.getElementById('i7s');
const L=50,Rr=540,T=15,Bt=255,YM=9,sx=n=>L+(Math.log(n)-Math.log(8))/(Math.log(313)-Math.log(8))*(Rr-L),sy=v=>Bt-Math.min(v,YM)/YM*(Bt-T);
function path(arr){{return arr.map((v,i)=>(i?'L':'M')+sx(D.n[i]).toFixed(1)+' '+sy(v).toFixed(1)).join(' ');}}
function draw(){{const n=Math.round(8*Math.exp(Math.log(313/8)*(+sl.value)/1000)),k=n-8;svg.innerHTML='';
 for(let v=0;v<=8;v+=2){{el(svg,'line',{{x1:L,x2:Rr,y1:sy(v),y2:sy(v),stroke:'#e3e3e3'}});el(svg,'text',{{x:L-8,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'}},v);}}
 for(const x of [10,20,50,100,200,300]){{el(svg,'text',{{x:sx(x),y:Bt+17,'text-anchor':'middle','font-size':12,fill:'#444'}},x);}}
 el(svg,'text',{{x:(L+Rr)/2,y:292,'text-anchor':'middle','font-size':13,fill:'#333'}},'Number of training cars n');
 el(svg,'text',{{x:14,y:(T+Bt)/2,'text-anchor':'middle','font-size':13,fill:'#333',transform:'rotate(-90 14 '+(T+Bt)/2+')'}},'RMSE (mpg)');
 el(svg,'line',{{x1:L,x2:Rr,y1:sy(D.base),y2:sy(D.base),stroke:'#777','stroke-dasharray':'5 4'}});el(svg,'text',{{x:Rr,y:sy(D.base)-5,'text-anchor':'end','font-size':11,fill:'#666'}},'Baseline '+D.base.toFixed(2));
 el(svg,'path',{{d:path(D.test),fill:'none',stroke:'{R}','stroke-width':2.6}});
 el(svg,'path',{{d:path(D.train),fill:'none',stroke:'{R}','stroke-width':2,'stroke-dasharray':'6 4',opacity:.75}});
 el(svg,'text',{{x:Rr,y:sy(D.test[150-8])-8,'text-anchor':'end','font-size':12,fill:'{R}','font-weight':700}},'Test RMSE');
 el(svg,'text',{{x:Rr,y:sy(D.train[150-8])+18,'text-anchor':'end','font-size':12,fill:'{R}'}},'Training RMSE (dashed)');
 el(svg,'line',{{x1:sx(n),x2:sx(n),y1:T,y2:Bt,stroke:'#222','stroke-width':1}});
 el(svg,'circle',{{cx:sx(n),cy:sy(D.test[k]),r:5,fill:'{R}'}});el(svg,'circle',{{cx:sx(n),cy:sy(D.train[k]),r:5,fill:'#fff',stroke:'{R}','stroke-width':2}});
 document.getElementById('i7n').textContent=n;document.getElementById('i7a').textContent=D.train[k].toFixed(2);
 const t=D.test[k];document.getElementById('i7b').textContent=t>=30?'above 30':t.toFixed(2);}}
sl.addEventListener('input',draw);draw();}})();</script>""".replace("__D__", json.dumps({k: D["I7"][k] for k in ["n", "train", "test", "base"]})))

# ---------------- I5adv ----------------
wrap("I5adv", f"""<div class="wbox" id="w-g">
<canvas width="560" height="360" style="max-width:560px"></canvas>
<div class="wrow"><label>Learning rate η <input type="range" id="gE" min="0.01" max="0.60" step="0.01" value="0.10" style="width:200px"> <span class="wn" id="gEv"></span></label>
<label>Steps <input type="range" id="gS" min="0" max="300" step="1" value="30" style="width:200px"> <span class="wn" id="gSv"></span></label></div>
<div class="wrow"><span id="gMsg"></span></div>
<div class="cap">Standardised hp and weight features. The horizontal axis is the coefficient of hp and the vertical axis is the coefficient of weight. Darker means larger MSE. The red dot is the least squares solution and the green path is gradient descent starting from (0, 0).</div></div>
<script>(function(){{
const D=__D__;const cv=document.querySelector('#w-g canvas'),ctx=cv.getContext('2d');
const W=560,Hh=360,L=55,Rr=545,T=10,Bt=320,X0=-5,X1=1.5,Y0=-8.5,Y1=1.5;
const px=x=>L+(x-X0)/(X1-X0)*(Rr-L),py=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
const H=D.H,ws=D.wstar;
function Jd(a,b){{const d0=a-ws[0],d1=b-ws[1];return .5*(H[0][0]*d0*d0+2*H[0][1]*d0*d1+H[1][1]*d1*d1);}}
const bg=document.createElement('canvas');bg.width=W;bg.height=Hh;const bc=bg.getContext('2d');
(function(){{const im=bc.createImageData(W,Hh);for(let j=0;j<Hh;j++)for(let i=0;i<W;i++){{
 if(i<L||i>Rr||j<T||j>Bt)continue;const a=X0+(i-L)/(Rr-L)*(X1-X0),b=Y1-(j-T)/(Bt-T)*(Y1-Y0);
 const d=Jd(a,b);let t=Math.max(0,Math.min(1,(Math.log10(d+0.02)+1.7)/3.2));
 t=Math.round(t*9)/9;const k=(j*W+i)*4;im.data[k]=255-(255-200)*t;im.data[k+1]=255-(255-75)*t;im.data[k+2]=255-(255-0)*t;im.data[k+3]=255;}}bc.putImageData(im,0,0);}})();
const eig=D.eig,lim=2/eig[1];
function run(eta,n){{let a=0,b=0;const P=[[a,b]];for(let s=0;s<n;s++){{const d0=a-ws[0],d1=b-ws[1];
 const g0=H[0][0]*d0+H[0][1]*d1,g1=H[0][1]*d0+H[1][1]*d1;a-=eta*g0;b-=eta*g1;P.push([a,b]);if(Math.abs(a)>1e6||Math.abs(b)>1e6)break;}}return P;}}
function draw(){{const eta=+document.getElementById('gE').value,n=+document.getElementById('gS').value;
 document.getElementById('gEv').textContent=eta.toFixed(2);document.getElementById('gSv').textContent=n;
 ctx.clearRect(0,0,W,Hh);ctx.drawImage(bg,0,0);ctx.strokeStyle='#888';ctx.fillStyle='#444';ctx.font='12px sans-serif';ctx.lineWidth=1;
 ctx.strokeRect(L,T,Rr-L,Bt-T);ctx.textAlign='center';
 for(let x=-4;x<=1;x+=1){{ctx.fillText(x,px(x),Bt+16);}}ctx.fillText('Coefficient of hp (standardised)',(L+Rr)/2,Bt+34);
 ctx.textAlign='right';for(let y=-8;y<=1;y+=2){{ctx.fillText(y,L-6,py(y)+4);}}
 ctx.save();ctx.translate(12,(T+Bt)/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('Coefficient of weight (standardised)',0,0);ctx.restore();
 const P=run(eta,n);ctx.save();ctx.beginPath();ctx.rect(L,T,Rr-L,Bt-T);ctx.clip();
 ctx.strokeStyle='{G}';ctx.lineWidth=2;ctx.beginPath();P.forEach((p,i)=>{{i?ctx.lineTo(px(p[0]),py(p[1])):ctx.moveTo(px(p[0]),py(p[1]));}});ctx.stroke();
 ctx.fillStyle='{G}';P.forEach((p,i)=>{{if(i%Math.max(1,Math.ceil(P.length/40))===0||i===P.length-1){{ctx.beginPath();ctx.arc(px(p[0]),py(p[1]),2.6,0,7);ctx.fill();}}}});
 ctx.fillStyle='#fff';ctx.strokeStyle='{G}';ctx.beginPath();ctx.arc(px(0),py(0),6,0,7);ctx.fill();ctx.stroke();
 ctx.fillStyle='{R}';ctx.beginPath();ctx.arc(px(ws[0]),py(ws[1]),6,0,7);ctx.fill();ctx.restore();
 const last=P[P.length-1],ex=Jd(last[0],last[1]);
 const msg=document.getElementById('gMsg');
 if(!isFinite(ex)||ex>1e4)msg.innerHTML='<b style="color:{R}">Diverged: the learning rate is too large, so the path jumps further away.</b>';
 else msg.innerHTML='Step '+(P.length-1)+', MSE is above the minimum by <b>'+ex.toFixed(ex<0.01?4:2)+'</b>';}}
document.getElementById('gE').addEventListener('input',draw);document.getElementById('gS').addEventListener('input',draw);draw();}})();</script>""".replace("__D__", json.dumps(D["I5adv"])))
print("widgets ok")
