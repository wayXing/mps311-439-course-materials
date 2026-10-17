"""Lesson 8 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I1  K-means step by step on three blobs: assign, move, repeat; two starting points
I2  three methods (K-means, single linkage, Ward) on two rings and two strips
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
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data)).replace("__EL__", EL) + "\n```\n"
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



COLS = ["#0072B2", "#00795A", "#C84B00"]
Bl = N["blobs"]
wrap("I1", """<div class="wbox" id="w-i1">
<div class="wrow" style="align-items:flex-start;gap:10px">
<svg id="i1s" viewBox="0 0 420 380" style="max-width:420px"></svg>
<div style="padding-top:20px;min-width:230px"><div>群内散布</div><div class="wn" id="i1v" style="font-size:2em"></div>
<div style="margin-top:12px">已做的步数：<span class="wn" id="i1n"></span></div><div id="i1msg" style="margin-top:12px;color:#555;max-width:230px"></div></div></div>
<div class="wrow"><button id="i1a">① 分给最近的中心</button><button id="i1b">② 中心挪到平均</button><button id="i1r">换个起点</button></div>
<div class="cap">十字是三个中心。先点 ①：每个点涂成离它最近的中心的颜色。再点 ②：每个中心挪到自己那群点的平均位置。交替点，看群内散布（每个点到自己中心的距离平方之和）怎样变。“换个起点”在两个起点之间切换。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.getElementById('i1s');const cols=__C__;
const L=34,Rr=410,T=8,Bt=345,X0=-2,X1=6.5,Y0=-2,Y1=5.5,sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
let which='A',C,lab,n,phase;
function reset(){C=D.starts[which].map(p=>p.slice());lab=null;n=0;phase='assign';draw();}
function inertia(){let s=0;D.x.forEach((x,i)=>{const y=D.y[i];let b=1e9;C.forEach(c=>{const d=(x-c[0])**2+(y-c[1])**2;if(d<b)b=d;});s+=b;});return s;}
function assign(){lab=D.x.map((x,i)=>{const y=D.y[i];let b=1e9,k=0;C.forEach((c,j)=>{const d=(x-c[0])**2+(y-c[1])**2;if(d<b){b=d;k=j;}});return k;});n++;draw();}
function move(){if(!lab)return;let changed=false;C=C.map((c,j)=>{let sx_=0,sy_=0,m=0;D.x.forEach((x,i)=>{if(lab[i]===j){sx_+=x;sy_+=D.y[i];m++;}});
 if(!m)return c;const nc=[sx_/m,sy_/m];if(Math.abs(nc[0]-c[0])+Math.abs(nc[1]-c[1])>1e-6)changed=true;return nc;});n++;
 document.getElementById('i1msg').textContent=changed?'':'中心不再移动：停下了。';draw();}
function draw(){svg.innerHTML='';el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 [0,2,4,6].forEach(v=>el(svg,'text',{x:sx(v),y:Bt+16,'text-anchor':'middle','font-size':12,fill:'#444'},v));[0,2,4].forEach(v=>el(svg,'text',{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v));
 D.x.forEach((x,i)=>el(svg,'circle',{cx:sx(x),cy:sy(D.y[i]),r:4,fill:lab?cols[lab[i]]:'#999','fill-opacity':0.75}));
 C.forEach((c,j)=>{el(svg,'text',{x:sx(c[0]),y:sy(c[1])+9,'text-anchor':'middle','font-size':30,'font-weight':700,fill:cols[j],stroke:'#fff','stroke-width':1.5},'+');});
 document.getElementById('i1v').textContent=inertia().toFixed(1);document.getElementById('i1n').textContent=n;}
document.getElementById('i1a').addEventListener('click',()=>{document.getElementById('i1msg').textContent='';assign();});
document.getElementById('i1b').addEventListener('click',move);
document.getElementById('i1r').addEventListener('click',()=>{which=which==='A'?'B':'A';document.getElementById('i1msg').textContent='起点 '+which;reset();});
reset();})();</script>""".replace("__C__", json.dumps(COLS)),
     dict(x=Bl["x"], y=Bl["y"], starts={k: v["start"] for k, v in Bl["runs"].items()}))

Sh = N["shapes"]
wrap("I2", """<div class="wbox" id="w-i2">
<div class="wrow"><span>形状：</span><button data-s="rings">两个环</button><button data-s="strips">两条带</button>
<span style="margin-left:14px">方法：</span><button data-m="kmeans">K-means</button><button data-m="single">层次聚类：最近的两点</button><button data-m="ward">层次聚类：Ward</button></div>
<div class="wrow" style="align-items:flex-start;gap:10px"><svg id="i2s" viewBox="0 0 420 380" style="max-width:420px"></svg>
<div style="padding-top:30px;min-width:200px"><div>找回原来的两组</div><div class="wn" id="i2a" style="font-size:2em"></div></div></div>
<div class="cap">每种方法都分成 2 群，颜色是它分出的群。右边是找回的比例：每一群按它里面最多的那一组算对。“最近的两点”：两群里最近的两个点有多近；Ward：合并后群内散布增加多少。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.getElementById('i2s');let s='rings',m='kmeans';const cols=['#0072B2','#C84B00'];
const L=34,Rr=410,T=8,Bt=345,X0=-4.5,X1=4.5,Y0=-4,Y1=4,sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
function draw(){const d=D[s];svg.innerHTML='';el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 d.x.forEach((x,i)=>el(svg,'circle',{cx:sx(x),cy:sy(d.y[i]),r:3.6,fill:cols[d[m].labels[i]],'fill-opacity':0.8}));
 document.getElementById('i2a').textContent=(100*d[m].acc).toFixed(1)+'%';
 document.querySelectorAll('#w-i2 button').forEach(b=>{const on=(b.dataset.s===s)||(b.dataset.m===m);b.style.borderColor=on?'""" + G + """':'#ccc';b.style.fontWeight=on?'700':'400';});}
document.querySelectorAll('#w-i2 button').forEach(b=>b.addEventListener('click',()=>{if(b.dataset.s)s=b.dataset.s;if(b.dataset.m)m=b.dataset.m;draw();}));draw();})();</script>""",
     {k: dict(x=v["x"], y=v["y"], kmeans=v["kmeans"], single=v["single"], ward=v["ward"]) for k, v in Sh.items()})
# two variants of I2 (different element ids, so both can sit on one page):
# I2k  K-means only, for the section where K-means fails (no hint of the other methods yet)
# I2b  all three methods, starting from nearest points, for the hierarchical clustering section
html2 = open(os.path.join(OUTDIR, "I2.html"), encoding="utf8").read()
k_only = html2
for a, b in [("w-i2", "w-i2k"), ("i2s", "i2ks"), ("i2a", "i2ka")]:
    k_only = k_only.replace(a, b)
k_only = re.sub(r'<span style="margin-left:14px">.*?</button></div>', "</div>", k_only, count=1, flags=re.S)
k_only = re.sub(r'(<div class="cap">)[^<]*(</div>)', r"\1" + ("K-means splits the points into 2 groups; the colours are its groups. On the right is the share recovered: each group counts as right for its most common true group." if EN else "K-means 把点分成 2 群，颜色是它分出的群。右边是找回的比例：每一群按它里面最多的那一组算对。") + r"\2", k_only, count=1)
open(os.path.join(OUTDIR, "I2k.html"), "w", encoding="utf8").write(k_only)
both = html2
for a, b in [("w-i2", "w-i2b"), ("i2s", "i2bs"), ("i2a", "i2ba"), ("let s='rings',m='kmeans'", "let s='rings',m='single'")]:
    both = both.replace(a, b)
open(os.path.join(OUTDIR, "I2b.html"), "w", encoding="utf8").write(both)
print("widgets ok")
