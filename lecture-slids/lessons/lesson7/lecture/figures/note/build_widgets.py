"""Lesson 7 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I1  turn a line through a 2-D cloud: spread of the projections against the reconstruction error
I2  rebuild one digit from k components: drag k and watch the image and the variance kept
"""
import json, os, re, subprocess, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
EN = "en" in " ".join(os.sys.argv[1:]).split()  # also accepts "--lang en" passed as one word
OUTDIR = os.path.join(HERE, "..", "note_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
N = json.load(open(os.path.join(HERE, "numbers.json")))
B, G, R = "#0072B2", "#00795A", "#C84B00"
CSS = """<style>
.wbox{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:1.2em 0;background:#fafbfc;font-size:.95em}
.wbox svg{display:block;max-width:100%;height:auto}
.wbox .wrow{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin-top:8px}
.wbox .wn{font-variant-numeric:tabular-nums;font-weight:700}
.wbox .cap{color:#555;font-size:.9em;margin-top:6px}
.wbox input[type=range]{width:220px}
.wbox button{font:inherit;padding:3px 10px;border:2px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
</style>"""
EL = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}"""


def wrap(name, body, data):
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data, ensure_ascii=False)).replace("__EL__", EL) + "\n```\n"
    if EN:
        from i18n_en import tr, check
        html = check(tr(html), name)
    open(os.path.join(OUTDIR, f"{name}.html"), "w", encoding="utf8").write(html)
    if shutil.which("node"):  # a translated apostrophe can break a single-quoted JS string
        js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        res = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True)
        os.remove(fh.name)
        if res.returncode:
            raise SystemExit(f"JavaScript syntax error in {name}: {res.stderr[:300]}")


C = N["cloud"]
wrap("I1", """<div class="wbox" id="w-i1">
<div class="wrow" style="align-items:flex-start;gap:10px">
<svg id="i1s" viewBox="0 0 420 420" style="max-width:420px"></svg>
<svg id="i1b" viewBox="0 0 260 300" style="max-width:260px"></svg></div>
<div class="wrow"><label>线的方向 <input type="range" id="i1a" min="0" max="179" step="1" value="0"> <span class="wn" id="i1av"></span></label>
<button id="i1best">散得最开的方向</button></div>
<div class="cap">200 个点，两个特征。绿线是一条过中心的线，每个点沿灰色短线落到绿线上（投影）。右边两根柱子：投影后的点沿绿线散得多开（蓝），以及把点换成它的投影后丢掉的距离平方的平均（红，还原误差）。转动绿线，看两根柱子怎样变。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.getElementById('i1s'),bar=document.getElementById('i1b'),sl=document.getElementById('i1a');
const n=D.x.length;let mx=0,my=0;D.x.forEach((v,k)=>{mx+=v;my+=D.y[k];});mx/=n;my/=n;
const L=30,Rr=410,T=10,Bt=390,X0=-5.5,X1=5.5,sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-X0)/(X1-X0)*(Bt-T);
function draw(){const a=+sl.value,ux=Math.cos(a*Math.PI/180),uy=Math.sin(a*Math.PI/180);svg.innerHTML='';
 el(svg,'line',{x1:L,x2:Rr,y1:sy(0),y2:sy(0),stroke:'#ddd'});el(svg,'line',{x1:sx(0),x2:sx(0),y1:T,y2:Bt,stroke:'#ddd'});
 D.x.forEach((x,k)=>{const cx=x-mx,cy=D.y[k]-my,z=cx*ux+cy*uy,px=z*ux,py=z*uy;
  el(svg,'line',{x1:sx(cx),y1:sy(cy),x2:sx(px),y2:sy(py),stroke:'#bbb','stroke-width':1});});
 D.x.forEach((x,k)=>{const cx=x-mx,cy=D.y[k]-my;el(svg,'circle',{cx:sx(cx),cy:sy(cy),r:3.4,fill:'""" + B + """','fill-opacity':0.75});});
 el(svg,'line',{x1:sx(-7*ux),y1:sy(-7*uy),x2:sx(7*ux),y2:sy(7*uy),stroke:'""" + G + """','stroke-width':3});
 const s=D.spread[a],e=D.error[a],tot=D.total;bar.innerHTML='';const H=230,base=262;
 [[s,'""" + B + """','散布',40],[e,'""" + R + """','还原误差',150]].forEach(([v,c,lab,x])=>{const h=H*v/tot;
  el(bar,'rect',{x:x,y:base-h,width:70,height:h,fill:c,'fill-opacity':0.85});el(bar,'text',{x:x+35,y:base-h-8,'text-anchor':'middle','font-size':15,'font-weight':700,fill:c},v.toFixed(2));
  el(bar,'text',{x:x+35,y:base+20,'text-anchor':'middle','font-size':14,fill:'#333'},lab);});
 el(bar,'line',{x1:20,x2:240,y1:base,y2:base,stroke:'#888'});
 el(bar,'text',{x:130,y:20,'text-anchor':'middle','font-size':14,fill:'#333'},'两者之和 = '+tot.toFixed(2));
 el(bar,'text',{x:130,y:40,'text-anchor':'middle','font-size':14,fill:'""" + B + """'},'保住 '+Math.round(100*s/tot)+'%');
 document.getElementById('i1av').textContent=a+'°';}
document.getElementById('i1best').addEventListener('click',()=>{sl.value=D.best;draw();});sl.addEventListener('input',draw);draw();})();</script>""",
     dict(x=C["x"], y=C["y"], spread=C["spread"], error=C["error"], total=C["total"], best=C["best_angle"]))

D = N["digits"]
wrap("I2", """<div class="wbox" id="w-i2">
<div class="wrow" style="align-items:flex-start;gap:18px">
<div><div style="text-align:center;margin-bottom:4px">原图</div><svg id="i2o" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div><div style="text-align:center;margin-bottom:4px">用 <span class="wn" id="i2kv"></span> 个主成分还原</div><svg id="i2r" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div><div style="text-align:center;margin-bottom:4px">原图 − 还原（误差）</div><svg id="i2e" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div style="padding-top:24px;min-width:200px"><div>这张图的重建误差：<span class="wn" id="i2err" style="font-size:1.4em"></span></div>
<div style="margin-top:8px">保住的总方差：<span class="wn" id="i2c"></span></div><div style="margin-top:8px">每张图要存的数：<span class="wn" id="i2n"></span>（原来 784）</div>
<div style="margin-top:8px;font-size:.88em;color:#555">误差图：红 = 还原得太白，蓝 = 还原得太黑</div></div></div>
<div class="wrow"><label>主成分个数 k <input type="range" id="i2k" min="1" max="150" step="1" value="10" style="width:320px"></label></div>
<div class="wrow"><span>换一张：</span><button data-i="0">图 1</button><button data-i="1">图 2</button><button data-i="2">图 3</button><button data-i="3">图 4</button></div>
<div class="cap">一张没参与拟合的验证图。还原 = 平均图 + 每个主成分乘上这张图在它上面的得分，只加前 k 个。重建误差是 784 个像素上“原图 − 还原”的平方和。拖动 k，看从几个开始认得出是几，误差怎样变小。</div></div>
<script>(function(){
__EL__
const D=__D__;const so=document.getElementById('i2o'),sr=document.getElementById('i2r'),se=document.getElementById('i2e'),sl=document.getElementById('i2k');let im=0;const c=6;
function paint(svg,v){svg.innerHTML='';for(let i=0;i<28;i++)for(let j=0;j<28;j++){const g=Math.max(0,Math.min(1,v[i*28+j]));const q=Math.round(255*(1-g));
 el(svg,'rect',{x:j*c,y:i*c,width:c+.4,height:c+.4,fill:'rgb('+q+','+q+','+q+')'});}el(svg,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#bbb'});}
function paintErr(svg,e){svg.innerHTML='';for(let i=0;i<28;i++)for(let j=0;j<28;j++){const v=Math.max(-1,Math.min(1,e[i*28+j]));let col;
 {const t=Math.min(1,Math.abs(v)),e=v>=0?[178,24,43]:[33,102,172];col='rgb('+e.map(x=>Math.round(255+(x-255)*t)).join(',')+')';}
 el(svg,'rect',{x:j*c,y:i*c,width:c+.4,height:c+.4,fill:col});}el(svg,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#bbb'});}
function draw(){const k=+sl.value,z=D.z[im],rec=D.mean.slice();for(let q=0;q<k;q++)for(let p=0;p<784;p++)rec[p]+=z[q]*D.comps[q][p];
 const img=D.imgs[im],err=img.map((v,p)=>v-rec[p]);let ss=0;err.forEach(v=>{ss+=v*v;});
 paint(so,img);paint(sr,rec);paintErr(se,err);document.getElementById('i2kv').textContent=k;document.getElementById('i2c').textContent=(100*D.cum[k-1]).toFixed(1)+'%';
 document.getElementById('i2n').textContent=k;document.getElementById('i2err').textContent=ss.toFixed(1);
 document.querySelectorAll('#w-i2 button').forEach(b=>{b.style.borderColor=(+b.dataset.i===im)?'""" + G + """':'#ccc';b.style.fontWeight=(+b.dataset.i===im)?'700':'400';});}
document.querySelectorAll('#w-i2 button').forEach(b=>b.addEventListener('click',()=>{im=+b.dataset.i;draw();}));sl.addEventListener('input',draw);draw();})();</script>""",
     dict(mean=D["mean"], comps=D["all_comps"], cum=D["cum"], imgs=[D["original"]] + D["extra_imgs"], z=[D["img_z"]] + D["extra_z"]))
# ---------------- I3: a second direction: does it repeat the first? ----------------
wrap("I3", """<div class="wbox" id="w-i3">
<div class="wrow" style="align-items:flex-start;gap:10px">
<svg id="i3s" viewBox="0 0 420 420" style="max-width:420px"></svg>
<div style="padding-top:40px;min-width:240px"><div>与第一个数的相关系数</div><div class="wn" id="i3c" style="font-size:2em"></div>
<div style="margin-top:14px">沿第二条线的散布</div><div class="wn" id="i3v" style="font-size:1.6em"></div></div></div>
<div class="wrow"><label>第二条线的方向 <input type="range" id="i3a" min="0" max="179" step="1" value="76"> <span class="wn" id="i3av"></span></label></div>
<div class="cap">绿线固定在散得最开的方向（第一个数）。橙线是第二个方向，拖动它。右边是两个数（点在绿线上的位置、在橙线上的位置）的相关系数：越接近 ±1，第二个数越是在重复第一个。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.getElementById('i3s'),sl=document.getElementById('i3a');
const n=D.x.length;let mx=0,my=0;D.x.forEach((v,k)=>{mx+=v;my+=D.y[k];});mx/=n;my/=n;
const L=30,Rr=410,T=10,Bt=390,X0=-5.5,X1=5.5,sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-X0)/(X1-X0)*(Bt-T);
const a1=D.best*Math.PI/180,u1=[Math.cos(a1),Math.sin(a1)];
function draw(){const a=+sl.value*Math.PI/180,u=[Math.cos(a),Math.sin(a)];svg.innerHTML='';
 let s11=0,s22=0,s12=0;D.x.forEach((x,k)=>{const cx=x-mx,cy=D.y[k]-my,z1=cx*u1[0]+cy*u1[1],z2=cx*u[0]+cy*u[1];s11+=z1*z1;s22+=z2*z2;s12+=z1*z2;
  el(svg,'circle',{cx:sx(cx),cy:sy(cy),r:3.2,fill:'""" + B + """','fill-opacity':0.6});});
 el(svg,'line',{x1:sx(-7*u1[0]),y1:sy(-7*u1[1]),x2:sx(7*u1[0]),y2:sy(7*u1[1]),stroke:'""" + G + """','stroke-width':3});
 el(svg,'line',{x1:sx(-7*u[0]),y1:sy(-7*u[1]),x2:sx(7*u[0]),y2:sy(7*u[1]),stroke:'#E69F00','stroke-width':3});
 const c=s12/Math.sqrt(s11*s22);document.getElementById('i3c').textContent=c.toFixed(2);document.getElementById('i3v').textContent=(s22/n).toFixed(2);
 document.getElementById('i3av').textContent=sl.value+'°';}
sl.addEventListener('input',draw);draw();})();</script>""", dict(x=C["x"], y=C["y"], best=N["cloud"]["pc_angle"]))

# ---------------- I4: keep the k most variable pixels ----------------
wrap("I4", """<div class="wbox" id="w-i4">
<div class="wrow" style="align-items:flex-start;gap:24px">
<div><div style="text-align:center;margin-bottom:4px">原图</div><svg id="i4o" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div><div style="text-align:center;margin-bottom:4px">只留 <span class="wn" id="i4kv"></span> 个像素（绿框），其余填平均值</div><svg id="i4r" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div style="padding-top:30px"><div>保住的总方差：<span class="wn" id="i4c" style="font-size:1.6em"></span></div></div></div>
<div class="wrow"><label>留下的像素个数 <input type="range" id="i4k" min="1" max="200" step="1" value="10" style="width:320px"></label></div>
<div class="cap">按散得开的程度，从大到小留下前几个像素，其余像素用它们在拟合图像里的平均值填上。拖动个数，看保住的方差和图像怎样变。</div></div>
<script>(function(){
__EL__
const D=__D__;const so=document.getElementById('i4o'),sr=document.getElementById('i4r'),sl=document.getElementById('i4k');const c=6;
function paint(svg,v,keep){svg.innerHTML='';for(let p=0;p<784;p++){const i=Math.floor(p/28),j=p%28,g=Math.max(0,Math.min(1,v[p])),q=Math.round(255*(1-g));
 el(svg,'rect',{x:j*c,y:i*c,width:c+.4,height:c+.4,fill:'rgb('+q+','+q+','+q+')'});}
 if(keep)keep.forEach(p=>{const i=Math.floor(p/28),j=p%28;el(svg,'rect',{x:j*c+.5,y:i*c+.5,width:c-1,height:c-1,fill:'none',stroke:'""" + G + """','stroke-width':1.2});});
 el(svg,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#bbb'});}
function draw(){const k=+sl.value,keep=new Set(D.order.slice(0,k)),rec=D.mean.map((m,p)=>keep.has(p)?D.img[p]:m);
 paint(so,D.img,null);paint(sr,rec,keep);document.getElementById('i4kv').textContent=k;document.getElementById('i4c').textContent=(100*D.cum[k-1]).toFixed(1)+'%';}
sl.addEventListener('input',draw);draw();})();</script>""", dict(order=D["pix_order"][:200], cum=D["pix_cum"][:200], mean=D["mean"], img=D["original"]))
print("widgets ok")
