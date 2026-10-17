"""Lesson 6 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I1  write your own rule on the ring: drag four cut lines to make a box
I2  one node, one feature: drag the cut and watch the error rate (flat) and Gini (dips)
I3  a tree on two features of the lumps: drag the depth, watch the rectangles and two accuracies
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
.wbox input[type=range]{width:180px}
</style>"""
EL = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}
function dot(svg,x,y,c,wrong){el(svg,'circle',c?{cx:x,cy:y,r:4.2,fill:'""" + B + """','fill-opacity':0.9}:{cx:x,cy:y,r:3.4,fill:'none',stroke:'#555','stroke-width':1.2});
 if(wrong)el(svg,'circle',{cx:x,cy:y,r:7.2,fill:'none',stroke:'""" + R + """','stroke-width':1.3});}"""


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


# ---------------- I1: write a rule yourself on the ring ----------------
Rg = N["ring"]
wrap("I1", """<div class="wbox" id="w-i1">
<svg viewBox="0 0 420 420" style="max-width:420px"></svg>
<div class="wrow"><label>左 <input type="range" id="i1a" min="-3.5" max="3.5" step="0.1" value="-3.4"></label><label>右 <input type="range" id="i1b" min="-3.5" max="3.5" step="0.1" value="3.4"></label></div>
<div class="wrow"><label>下 <input type="range" id="i1c" min="-3.5" max="3.5" step="0.1" value="-3.4"></label><label>上 <input type="range" id="i1d" min="-3.5" max="3.5" step="0.1" value="3.4"></label></div>
<div class="wrow" style="font-size:.9em;color:#444"><span><span style="color:#0072B2">●</span> 类 1（里圈）</span><span><span style="color:#777">○</span> 类 0（外圈）</span><span><span style="color:#C84B00;font-size:1.2em">◯</span> 判错</span></div>
<div class="wrow"><span id="i1rule"></span></div>
<div class="wrow"><span>准确率 <span class="wn" id="i1acc"></span></span><span>直线（logistic regression）：<span class="wn" id="i1line"></span></span></div>
<div class="cap">Lesson 4 的环形数据：蓝点是类 1（里圈），灰色空心圆是类 0（外圈）。拖四个滑块移动四条切线，方框里的判为类 1，判错的点画橙圈。你能写出一条准确率超过 0.95 的规则吗？</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.querySelector('#w-i1 svg');const L=36,Rr=410,T=8,Bt=382,X0=-3.5,X1=3.5;
const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-X0)/(X1-X0)*(Bt-T);const ids=['i1a','i1b','i1c','i1d'].map(i=>document.getElementById(i));
function draw(){let [a,b,c,d]=ids.map(e=>+e.value);if(a>b)[a,b]=[b,a];if(c>d)[c,d]=[d,c];svg.innerHTML='';
 el(svg,'rect',{x:sx(a),y:sy(d),width:sx(b)-sx(a),height:sy(c)-sy(d),fill:'""" + G + """','fill-opacity':0.15,stroke:'""" + G + """','stroke-width':2.6});
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 [-3,-2,-1,0,1,2,3].forEach(v=>{el(svg,'text',{x:sx(v),y:Bt+15,'text-anchor':'middle','font-size':12,fill:'#444'},v);el(svg,'text',{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v);});
 let ok=0;D.x.forEach((x,k)=>{const y=D.y[k],inside=x>a&&x<b&&y>c&&y<d,w=inside!==(D.label[k]===1);if(!w)ok++;dot(svg,sx(x),sy(y),D.label[k],w);});
 document.getElementById('i1acc').textContent=(ok/D.x.length).toFixed(3);
 document.getElementById('i1rule').textContent='如果 '+a.toFixed(1)+' < x₁ < '+b.toFixed(1)+' 且 '+c.toFixed(1)+' < x₂ < '+d.toFixed(1)+'，判为类 1；否则判为类 0';}
document.getElementById('i1line').textContent=D.line.toFixed(2);ids.forEach(e=>e.addEventListener('input',draw));draw();})();</script>""",
     dict(x=Rg["x"], y=Rg["y"], label=Rg["label"], line=Rg["acc_line"]))

# ---------------- I2: one node: error rate against Gini ----------------
Nd = N["node"]
wrap("I2", """<div class="wbox" id="w-i2">
<svg viewBox="0 0 560 360" style="max-width:560px"></svg>
<div class="wrow"><label>切在 <input type="range" id="i2c" min="0" max="38" step="1" value="8"> <span class="wn" id="i2cv"></span></label></div>
<div class="wrow"><span>左边：<span class="wn" id="i2l"></span></span><span>右边：<span class="wn" id="i2r"></span></span></div>
<div class="wrow"><span style="color:#555">错分率 <span class="wn" id="i2e"></span></span><span style="color:""" + G + """">Gini <span class="wn" id="i2g"></span></span></div>
<div class="cap">一个节点里有 100 个点：20 个蓝点挤在中间一段，80 个灰点散在各处。上面是这些点（上下错开以免重叠），竖线是切的位置。下面两条曲线：每个切点的错分率（两边各按多数类判，错的比例）和 Gini（两边各自的 Gini，按点数加权平均）。拖动切线，看哪条曲线会动。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.querySelector('#w-i2 svg'),sl=document.getElementById('i2c');sl.max=D.cuts.length-1;
const L=50,Rr=545,sx=x=>L+x/10*(Rr-L);const T2=170,B2=320,Y0=0.15,Y1=0.36,sy=v=>B2-(v-Y0)/(Y1-Y0)*(B2-T2);
function draw(){const k=+sl.value,c=D.cuts[k];svg.innerHTML='';
 D.x.forEach((x,i)=>{const yy=30+((i*7919)%97)/97*90;dot(svg,sx(x),yy,D.label[i],false);});
 el(svg,'line',{x1:sx(c),x2:sx(c),y1:18,y2:B2,stroke:'#333','stroke-width':2.2});
 el(svg,'line',{x1:L,x2:Rr,y1:B2,y2:B2,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T2,y2:B2,stroke:'#888'});
 [0,2,4,6,8,10].forEach(v=>el(svg,'text',{x:sx(v),y:B2+16,'text-anchor':'middle','font-size':12,fill:'#444'},v));
 [0.2,0.25,0.3,0.35].forEach(v=>el(svg,'text',{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v.toFixed(2)));
 el(svg,'text',{x:(L+Rr)/2,y:B2+34,'text-anchor':'middle','font-size':13,fill:'#333'},'切点的位置');
 let de='',dg='';D.cuts.forEach((cc,i)=>{de+=(i?'L':'M')+sx(cc).toFixed(1)+' '+sy(D.err[i]).toFixed(1);dg+=(i?'L':'M')+sx(cc).toFixed(1)+' '+sy(D.gini[i]).toFixed(1);});
 el(svg,'path',{d:de,fill:'none',stroke:'#777','stroke-width':3,'stroke-dasharray':'7 4'});el(svg,'path',{d:dg,fill:'none',stroke:'""" + G + """','stroke-width':3});
 el(svg,'circle',{cx:sx(c),cy:sy(D.err[k]),r:6,fill:'#777'});el(svg,'circle',{cx:sx(c),cy:sy(D.gini[k]),r:6,fill:'""" + G + """'});
 el(svg,'text',{x:Rr-4,y:sy(0.2)-8,'text-anchor':'end','font-size':13,fill:'#555'},'错分率');el(svg,'text',{x:Rr-4,y:sy(D.gini[D.gini.length-1])-10,'text-anchor':'end','font-size':13,fill:'""" + G + """'},'Gini');
 let lb=0,lg=0,rb=0,rg=0;D.x.forEach((x,i)=>{if(x<=c){D.label[i]?lb++:lg++;}else{D.label[i]?rb++:rg++;}});
 document.getElementById('i2cv').textContent=c.toFixed(2);document.getElementById('i2l').textContent=lb+' 蓝，'+lg+' 灰';document.getElementById('i2r').textContent=rb+' 蓝，'+rg+' 灰';
 document.getElementById('i2e').textContent=D.err[k].toFixed(3);document.getElementById('i2g').textContent=D.gini[k].toFixed(3);}
sl.addEventListener('input',draw);draw();})();</script>""",
     dict(x=Nd["x"], label=Nd["label"], cuts=Nd["cuts"], err=Nd["err"], gini=Nd["gini"]))

# ---------------- I3: depth of a tree on two features ----------------
C2 = N["bc2"]
wrap("I3", """<div class="wbox" id="w-i3">
<svg viewBox="0 0 470 420" style="max-width:470px"></svg>
<div class="wrow"><label>最大深度 <input type="range" id="i3d" min="1" max="10" step="1" value="1"> <span class="wn" id="i3dv"></span></label>
<label><input type="checkbox" id="i3v"> 显示验证集的点</label></div>
<div class="wrow"><span>叶子数 <span class="wn" id="i3n"></span></span><span style="color:""" + B + """">训练准确率 <span class="wn" id="i3t"></span></span><span style="color:""" + G + """">验证集上的准确率 <span class="wn" id="i3a"></span></span></div>
<div class="cap">两个特征：细胞核平均半径（横轴）和平均纹理（纵轴）。蓝色区域判为恶性。点是 319 个用来拟合的肿块：蓝点恶性，灰圈良性；勾选后换成验证集的 107 个肿块：它们是从训练集里分出的，没有参与拟合。拖动深度，看矩形怎样越切越碎，两个准确率怎样变。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.querySelector('#w-i3 svg'),sl=document.getElementById('i3d'),cb=document.getElementById('i3v');
const L=46,Rr=460,T=8,Bt=372,X0=6,X1=29,Y0=9,Y1=40;const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
function draw(){const dp=String(sl.value),g=D.grids[dp],nx=D.gx.length,ny=D.gy.length,dx=(X1-X0)/(nx-1),dy=(Y1-Y0)/(ny-1);svg.innerHTML='';
 const grp=el(svg,'g',{opacity:0.22});for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){if(g[j*nx+i]==='1')el(grp,'rect',{x:sx(D.gx[i]-dx/2),y:sy(D.gy[j]+dy/2),width:sx(dx)-sx(0)+0.6,height:sy(0)-sy(dy)+0.6,fill:'""" + B + """'});}
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 [10,15,20,25].forEach(v=>el(svg,'text',{x:sx(v),y:Bt+15,'text-anchor':'middle','font-size':12,fill:'#444'},v));[10,20,30,40].forEach(v=>el(svg,'text',{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v));
 el(svg,'text',{x:(L+Rr)/2,y:Bt+34,'text-anchor':'middle','font-size':13,fill:'#333'},'细胞核平均半径');el(svg,'text',{x:12,y:(T+Bt)/2,'font-size':13,fill:'#333',transform:'rotate(-90 12 '+(T+Bt)/2+')','text-anchor':'middle'},'细胞核平均纹理');
 const xs=cb.checked?D.vx:D.tx,ys=cb.checked?D.vy:D.ty,ls=cb.checked?D.vl:D.tl;
 xs.forEach((x,k)=>{if(x>=X0&&x<=X1&&ys[k]>=Y0&&ys[k]<=Y1)dot(svg,sx(x),sy(ys[k]),ls[k],false);});
 document.getElementById('i3dv').textContent=dp;document.getElementById('i3n').textContent=D.curve[dp].leaves;
 document.getElementById('i3t').textContent=D.curve[dp].train.toFixed(3);document.getElementById('i3a').textContent=D.curve[dp].val.toFixed(3);}
sl.addEventListener('input',draw);cb.addEventListener('change',draw);draw();})();</script>""",
     dict(grids=C2["grids"], gx=C2["gx"], gy=C2["gy"], tx=C2["train_x"], ty=C2["train_y"], tl=C2["train_label"],
          vx=C2["val_x"], vy=C2["val_y"], vl=C2["val_label"], curve=C2["curve"]))
print("widgets ok")
