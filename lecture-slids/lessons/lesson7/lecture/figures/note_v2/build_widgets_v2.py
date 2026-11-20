"""Lesson 7 Note V2: self-contained HTML widgets (no CDN), data from numbers_v2.json. Chinese.
Run (from this folder):  python build_widgets_v2.py
NOTE_LANG=en writes the English version to ../note_en_v2/; without it, the Chinese version here, unchanged.
I1  turn a line through the height and arm-span cloud: spread of the projections against the reconstruction error
I2  rebuild one digit from K components: drag K, see the formula, the image, the error and the share kept
I3  (V7) compress one image on its own: its 28 rows are the samples; same four images as I2 (data: v7_extra.json from build_v7_extra.py)
"""
import json, os, re, subprocess, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers_v2.json")))
EN = os.environ.get("NOTE_LANG") == "en"   # NOTE_LANG=en: English labels, written to ../note_en_v2/
OUT = os.path.join(os.path.dirname(HERE), "note_en_v2") if EN else HERE
os.makedirs(OUT, exist_ok=True)


def T(zh, en):
    return en if EN else zh

B, G, R = "#0072B2", "#00795A", "#C84B00"
CSS = """<style>
.wbox{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:1.2em 0;background:#fafbfc;font-size:.95em}
.wbox svg{display:block;max-width:100%;height:auto}
.wbox .wrow{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin-top:8px}
.wbox .wn{font-variant-numeric:tabular-nums;font-weight:700}
.wbox .cap{color:#555;font-size:.9em;margin-top:6px}
.wbox input[type=range]{width:220px}
.wbox button{font:inherit;padding:3px 10px;border:2px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
.wbox .fx{font-family:Georgia,serif;font-size:1.02em;margin-top:6px}
</style>"""
EL = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}"""


def wrap(name, body, data):
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data, ensure_ascii=False)).replace("__EL__", EL) + "\n```\n"
    open(os.path.join(OUT, f"{name}.html"), "w", encoding="utf8").write(html)
    if shutil.which("node"):
        js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        res = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True)
        os.remove(fh.name)
        if res.returncode:
            raise SystemExit(f"JavaScript syntax error in {name}: {res.stderr[:300]}")


A = N["arm"]
wrap("I1", """<div class="wbox" id="w-i1">
<div class="wrow" style="align-items:flex-start;gap:10px">
<svg id="i1s" viewBox="0 0 440 440" style="max-width:420px"></svg>
<svg id="i1b" """ + T('viewBox="0 0 260 300"', 'viewBox="0 0 260 330"') + """ style="max-width:260px"></svg></div>
<div class="wrow"><label>""" + T('线的方向 θ', 'Direction of the line θ') + """ <input type="range" id="i1a" min="0" max="179" step="1" value="0"> <span class="wn" id="i1av"></span></label>
<button id="i1best">""" + T('散得最开的方向', 'Direction of widest spread') + """</button></div>
<div class="cap">""" + T('200 个人的身高和臂展。绿线过中心，每个点沿灰色短线落到绿线上（投影）。右边两根柱子：投影后的点沿绿线散得多开（蓝，散布），以及把点换成投影后丢掉的距离平方的平均（红，还原误差），单位 cm²。0° 就是只报身高。转动绿线，看两根柱子怎样变。', 'Height and arm span of 200 people. The green line passes through the centre, and each point drops along a short grey line onto it (its projection). The two bars on the right: how widely the projected points spread along the green line (blue, spread), and the average squared distance lost when each point is replaced by its projection (red, reconstruction error), both in cm². 0° means reporting height only. Turn the green line and watch the two bars change.') + """</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.getElementById('i1s'),bar=document.getElementById('i1b'),sl=document.getElementById('i1a');
const n=D.h.length,mx=D.m[0],my=D.m[1],W=24;
const L=46,Rr=430,T=10,Bt=394,sx=x=>L+(x-(mx-W))/(2*W)*(Rr-L),sy=y=>Bt-(y-(my-W))/(2*W)*(Bt-T);
function axes(){[150,160,170,180,190].forEach(v=>{el(svg,'text',{x:sx(v),y:Bt+16,'text-anchor':'middle','font-size':11,fill:'#666'},v);
  el(svg,'text',{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':11,fill:'#666'},v);});
 el(svg,'text',{x:(L+Rr)/2,y:Bt+34,'text-anchor':'middle','font-size':12,fill:'#333'},""" + T("'身高（cm）'", "'Height (cm)'") + """);
 el(svg,'text',{x:12,y:(T+Bt)/2,'text-anchor':'middle','font-size':12,fill:'#333',transform:'rotate(-90 12 '+((T+Bt)/2)+')'},""" + T("'臂展（cm）'", "'Arm span (cm)'") + """);
 el(svg,'rect',{x:L,y:T,width:Rr-L,height:Bt-T,fill:'none',stroke:'#ddd'});}
function draw(){const a=+sl.value,ux=Math.cos(a*Math.PI/180),uy=Math.sin(a*Math.PI/180);svg.innerHTML='';axes();
 D.h.forEach((x,k)=>{const cx=x-mx,cy=D.a[k]-my,z=cx*ux+cy*uy;
  el(svg,'line',{x1:sx(x),y1:sy(D.a[k]),x2:sx(mx+z*ux),y2:sy(my+z*uy),stroke:'#bbb','stroke-width':1});});
 D.h.forEach((x,k)=>{el(svg,'circle',{cx:sx(x),cy:sy(D.a[k]),r:3.4,fill:'""" + B + """','fill-opacity':0.75});});
 const t=W/Math.max(Math.abs(ux),Math.abs(uy));el(svg,'line',{x1:sx(mx-t*ux),y1:sy(my-t*uy),x2:sx(mx+t*ux),y2:sy(my+t*uy),stroke:'""" + G + """','stroke-width':3});
 const s=D.spread[a],e=D.error[a],tot=D.total;bar.innerHTML='';const H=200,base=262;
 [[s,'""" + B + """',""" + T("'散布'", "['Spread','(variance along','the line)']") + """,40],[e,'""" + R + """',""" + T("'还原误差'", "['Reconstruction','error']") + """,150]].forEach(([v,c,lab,x])=>{const h=H*v/tot;
  el(bar,'rect',{x:x,y:base-h,width:70,height:h,fill:c,'fill-opacity':0.85});el(bar,'text',{x:x+35,y:base-h-8,'text-anchor':'middle','font-size':15,'font-weight':700,fill:c},v.toFixed(2));
  el(bar,'text',{x:x+35,y:base+20,'text-anchor':'middle',""" + T("'font-size':14,fill:'#333'},lab);});", "'font-size':14,fill:'#333'},lab[0]);lab.slice(1).forEach((t,i)=>el(bar,'text',{x:x+35,y:base+37+15*i,'text-anchor':'middle','font-size':12,fill:'#555'},t));});") + """
 el(bar,'line',{x1:20,x2:240,y1:base,y2:base,stroke:'#888'});
 el(bar,'text',{x:130,y:20,'text-anchor':'middle','font-size':14,fill:'#333'},""" + T("'两者之和 = '", "'Sum of the two = '") + """+tot.toFixed(2));
 el(bar,'text',{x:130,y:40,'text-anchor':'middle','font-size':14,fill:'""" + B + """'},""" + T("'留住 '", "'Keeps '") + """+Math.round(100*s/tot)+'%');
 document.getElementById('i1av').textContent=a+'°';}
document.getElementById('i1best').addEventListener('click',()=>{sl.value=Math.round(D.best);draw();});sl.addEventListener('input',draw);draw();})();</script>""",
     dict(h=A["h"], a=A["a"], m=[sum(A["h"]) / len(A["h"]), sum(A["a"]) / len(A["a"])], spread=A["spread"], error=A["error"], total=A["total"], best=A["theta"]))

D = N["digits"]
wrap("I2", """<div class="wbox" id="w-i2">
<div class="fx">x̂ = x̄ + z₁u₁ + z₂u₂ + … + z<sub>K</sub>u<sub>K</sub>""" + T('\u3000\u3000这张图：', '&emsp;&emsp;This image: ') + """x̂ = x̄ <span id="i2f"></span></div>
<div class="wrow" style="align-items:flex-start;gap:18px">
<div><div style="text-align:center;margin-bottom:4px">""" + T('原图 x', 'Original x') + """</div><svg id="i2o" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div><div style="text-align:center;margin-bottom:4px">""" + T('用 K = <span class="wn" id="i2kv"></span> 个数还原 x̂', 'Rebuilt x̂ from K = <span class="wn" id="i2kv"></span> numbers') + """</div><svg id="i2r" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div><div style="text-align:center;margin-bottom:4px">""" + T('x − x̂（丢掉的部分）', 'x − x̂ (the part lost)') + """</div><svg id="i2e" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div style="padding-top:24px;min-width:200px"><div>""" + T('这张图的重建误差：', 'Reconstruction error of this image: ') + """<span class="wn" id="i2err" style="font-size:1.4em"></span></div>
<div style="margin-top:8px">""" + T('K 个主成分留住的总方差：', 'Cumulative share kept by K components: ') + """<span class="wn" id="i2c"></span></div><div style="margin-top:8px">""" + T('每张图要存的数：', 'Numbers stored per image: ') + """<span class="wn" id="i2n"></span>""" + T('（原来 784）', ' (originally 784)') + """</div>
<div style="margin-top:8px;font-size:.88em;color:#555">""" + T('右图：红 = 还原得太白，蓝 = 还原得太黑', 'Right: red = rebuilt too light, blue = rebuilt too dark') + """</div></div></div>
<div class="wrow">""" + T('<label>主成分个数 K ', '<label>Number of principal components K ') + """<input type="range" id="i2k" min="0" max="150" step="1" value="10" style="width:320px"></label></div>
<div class="wrow"><span>""" + T('换一张：', 'Another image: ') + """</span><button data-i="0">""" + T('图 1</button>', 'Image 1</button>') + """<button data-i="1">""" + T('图 2</button>', 'Image 2</button>') + """<button data-i="2">""" + T('图 3</button>', 'Image 3</button>') + """<button data-i="3">""" + T('图 4</button>', 'Image 4</button>') + """</div>
<div class="cap">""" + T('验证集里的图，没有参与学主成分。还原 = 平均图 x̄ 加上每个主成分 u<sub>j</sub> 乘以这张图的得分 z<sub>j</sub>，只加前 K 个。重建误差是 784 个像素上 x − x̂ 的平方和。K = 0 时只剩平均图。', "Images from the validation set, not used to learn the principal components. Rebuild = the mean image x̄ plus each principal component u<sub>j</sub> times this image's score z<sub>j</sub>, adding only the first K. The reconstruction error is the sum of squares of x − x̂ over the 784 pixels. At K = 0 only the mean image is left.") + """</div></div>
<script>(function(){
__EL__
const D=__D__;const so=document.getElementById('i2o'),sr=document.getElementById('i2r'),se=document.getElementById('i2e'),sl=document.getElementById('i2k');let im=0;const c=6;
function paint(svg,v){svg.innerHTML='';for(let i=0;i<28;i++)for(let j=0;j<28;j++){const g=Math.max(0,Math.min(1,v[i*28+j]));const q=Math.round(255*(1-g));
 el(svg,'rect',{x:j*c,y:i*c,width:c+.4,height:c+.4,fill:'rgb('+q+','+q+','+q+')'});}el(svg,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#bbb'});}
function paintErr(svg,e){svg.innerHTML='';for(let i=0;i<28;i++)for(let j=0;j<28;j++){const v=Math.max(-1,Math.min(1,e[i*28+j]));
 const t=Math.min(1,Math.abs(v)),b=v>=0?[178,24,43]:[33,102,172],col='rgb('+b.map(x=>Math.round(255+(x-255)*t)).join(',')+')';
 el(svg,'rect',{x:j*c,y:i*c,width:c+.4,height:c+.4,fill:col});}el(svg,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#bbb'});}
function sgn(v){return (v<0?'− ':'+ ')+Math.abs(v).toFixed(2);}
function draw(){const k=+sl.value,z=D.z[im],rec=D.mean.slice();for(let q=0;q<k;q++)for(let p=0;p<784;p++)rec[p]+=z[q]*D.comps[q][p];
 const img=D.imgs[im],err=img.map((v,p)=>v-rec[p]);let ss=0;err.forEach(v=>{ss+=v*v;});
 paint(so,img);paint(sr,rec);paintErr(se,err);document.getElementById('i2kv').textContent=k;
 document.getElementById('i2c').textContent=k?(100*D.cum[k-1]).toFixed(1)+'%':'0%';
 let f='';for(let q=0;q<Math.min(k,3);q++)f+=' '+sgn(z[q])+'·u'+'₁₂₃'[q];if(k>3)f+=""" + T("' + …（共 '+k+' 项）'", "' + … ('+k+' terms)'") + """;document.getElementById('i2f').textContent=f;
 document.getElementById('i2n').textContent=k;document.getElementById('i2err').textContent=ss.toFixed(1);
 document.querySelectorAll('#w-i2 button').forEach(b=>{b.style.borderColor=(+b.dataset.i===im)?'""" + G + """':'#ccc';b.style.fontWeight=(+b.dataset.i===im)?'700':'400';});}
document.querySelectorAll('#w-i2 button').forEach(b=>b.addEventListener('click',()=>{im=+b.dataset.i;draw();}));sl.addEventListener('input',draw);draw();})();</script>""",
     dict(mean=D["mean"], comps=D["comps"], cum=D["cum"], imgs=[D["original"]] + D["extra_imgs"], z=[D["z"]] + D["extra_z"]))

V = json.load(open(os.path.join(HERE, "v7_extra.json")))
wrap("I3", """<div class="wbox" id="w-i3">
<div class="fx">""" + T('一张图自己做 PCA：28 行是 28 个样本，每行 28 个数。这一行 ≈ 平均行 ', 'PCA on one image alone: its 28 rows are 28 samples of 28 numbers each. This row ≈ mean row ') + """+ z₁u₁ + … + z<sub>K</sub>u<sub>K</sub></div>
<div class="wrow" style="align-items:flex-start;gap:18px">
<div><div style="text-align:center;margin-bottom:4px">""" + T('原图</div>', 'Original</div>') + """<svg id="i3o" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div><div style="text-align:center;margin-bottom:4px">""" + T('每行用 K = <span class="wn" id="i3kv"></span> 个数还原', 'Each row rebuilt from K = <span class="wn" id="i3kv"></span> numbers') + """</div><svg id="i3r" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div><div style="text-align:center;margin-bottom:4px">""" + T('丢掉的部分</div>', 'The part lost</div>') + """<svg id="i3e" viewBox="0 0 168 168" style="width:168px"></svg></div>
<div style="padding-top:24px;min-width:220px"><div>""" + T('这张图的重建误差：', 'Reconstruction error of this image: ') + """<span class="wn" id="i3err" style="font-size:1.4em"></span></div>
<div style="margin-top:8px">""" + T('这张图要存的数：', 'Numbers stored for this image: ') + """<span class="wn" id="i3n"></span>""" + T('（原来 784）', ' (originally 784)') + """</div>
<div style="margin-top:4px;font-size:.88em;color:#555">""" + T('= 平均行 28 + 主成分 28×K + 每行得分 28×K', '= mean row 28 + components 28×K + row scores 28×K') + """</div></div></div>
<div class="wrow">""" + T('<label>每行的主成分个数 K ', '<label>Principal components per row K ') + """<input type="range" id="i3k" min="0" max="28" step="1" value="3" style="width:280px"></label></div>
<div class="wrow"><span>""" + T('换一张：', 'Another image: ') + """</span><button data-i="0">""" + T('图 1</button>', 'Image 1</button>') + """<button data-i="1">""" + T('图 2</button>', 'Image 2</button>') + """<button data-i="2">""" + T('图 3</button>', 'Image 3</button>') + """<button data-i="3">""" + T('图 4</button>', 'Image 4</button>') + """</div>
<div class="cap">""" + T('和上一个组件是同样的四张验证图。主成分只从这一张图的 28 行里学，每张图有自己的一套，要和得分一起存下来。重建误差是 784 个像素上差的平方和。K = 0 时每一行都换成这张图的平均行。', "The same four validation images as in the previous widget. The principal components are learnt only from the 28 rows of this one image, so each image has its own set, which must be stored along with its scores. The reconstruction error is the sum of squared differences over the 784 pixels. At K = 0 every row is replaced by this image's mean row.") + """</div></div>
<script>(function(){
__EL__
const D=__D__;const so=document.getElementById('i3o'),sr=document.getElementById('i3r'),se=document.getElementById('i3e'),sl=document.getElementById('i3k');let im=0;const c=6;
function paint(svg,v){svg.innerHTML='';for(let i=0;i<28;i++)for(let j=0;j<28;j++){const g=Math.max(0,Math.min(1,v[i*28+j]));const q=Math.round(255*(1-g));
 el(svg,'rect',{x:j*c,y:i*c,width:c+.4,height:c+.4,fill:'rgb('+q+','+q+','+q+')'});}el(svg,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#bbb'});}
function paintErr(svg,e){svg.innerHTML='';for(let i=0;i<28;i++)for(let j=0;j<28;j++){const v=Math.max(-1,Math.min(1,e[i*28+j]));
 const t=Math.min(1,Math.abs(v)),b=v>=0?[178,24,43]:[33,102,172],col='rgb('+b.map(x=>Math.round(255+(x-255)*t)).join(',')+')';
 el(svg,'rect',{x:j*c,y:i*c,width:c+.4,height:c+.4,fill:col});}el(svg,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#bbb'});}
function draw(){const k=+sl.value,P=D.imgs[im],img=D.orig[im],rec=[];
 for(let i=0;i<28;i++)for(let j=0;j<28;j++){let v=P.mean[j];for(let q=0;q<k;q++)v+=P.Z[i][q]*P.comps[q][j];rec.push(v);}
 const err=img.map((v,p)=>v-rec[p]);let ss=0;err.forEach(v=>{ss+=v*v;});
 paint(so,img);paint(sr,rec);paintErr(se,err);document.getElementById('i3kv').textContent=k;
 document.getElementById('i3n').textContent=28+56*k;document.getElementById('i3err').textContent=ss.toFixed(1);
 document.querySelectorAll('#w-i3 button').forEach(b=>{b.style.borderColor=(+b.dataset.i===im)?'""" + G + """':'#ccc';b.style.fontWeight=(+b.dataset.i===im)?'700':'400';});}
document.querySelectorAll('#w-i3 button').forEach(b=>b.addEventListener('click',()=>{im=+b.dataset.i;draw();}));sl.addEventListener('input',draw);draw();})();</script>""",
     dict(imgs=[dict(mean=I["mean"], comps=I["comps"], Z=I["Z"]) for I in V["imgs"]], orig=[D["original"]] + D["extra_imgs"]))
print("widgets ok")
