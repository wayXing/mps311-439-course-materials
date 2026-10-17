"""Lesson 9 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I1  place three lines by hand on the Lesson 4 ring; a point is called "inner" when it is on the inner side of every line
I2  the trained networks: 1 to 4 hidden units, with sigmoid, ReLU or no activation in the middle
I2a the same with sigmoid only (Section 3, before the activation is questioned)
I2n sigmoid or no activation, without the second bend (slides only)
"""
import json, os, re, subprocess, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
EN = "en" in " ".join(os.sys.argv[1:]).split()  # also accepts "--lang en" passed as one word
OUTDIR = os.path.join(HERE, "..", "note_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
N = json.load(open(os.path.join(HERE, "numbers.json")))
CSS = """<style>
.wbox{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:1.2em 0;background:#fafbfc;font-size:.95em}
.wbox svg{display:block;max-width:100%;height:auto}
.wbox .wrow{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin-top:8px}
.wbox .wn{font-variant-numeric:tabular-nums;font-weight:700}
.wbox .cap{color:#555;font-size:.9em;margin-top:6px}
.wbox input[type=range]{width:150px}
.wbox button{font:inherit;padding:3px 10px;border:2px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
.wbox button.on{border-color:#00795A;color:#00795A;font-weight:700}
</style>"""
# shared drawing code: the ring on a square plot, and a clipped straight line a*x + b*y + c = 0
EL = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}
const LIM=3.6,PX=360,OFF=10,sx=x=>OFF+(x+LIM)/(2*LIM)*PX,sy=y=>OFF+(LIM-y)/(2*LIM)*PX;
function shade(svg,inside){const g=el(svg,'g',{opacity:0.22,'shape-rendering':'crispEdges'});const n=60,st=2*LIM/n,w=PX/n;
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){const x=-LIM+(i+.5)*st,y=-LIM+(j+.5)*st;if(inside(x,y))el(g,'rect',{x:OFF+i*w,y:OFF+(n-1-j)*w,width:w+.6,height:w+.6,fill:'#0072B2'});}}
function points(svg,D){D.x.forEach((x,i)=>el(svg,'circle',{cx:sx(x),cy:sy(D.y[i]),r:3.3,fill:D.label[i]?'#0072B2':'#E69F00'}));
 el(svg,'rect',{x:OFF,y:OFF,width:PX,height:PX,fill:'none',stroke:'#888'});}
function line(svg,a,b,c,col){const P=[];const L=LIM;
 if(Math.abs(b)>1e-9){[-L,L].forEach(x=>{const y=-(a*x+c)/b;if(y>=-L&&y<=L)P.push([x,y]);});}
 if(Math.abs(a)>1e-9){[-L,L].forEach(y=>{const x=-(b*y+c)/a;if(x>=-L&&x<=L)P.push([x,y]);});}
 if(P.length>=2)el(svg,'line',{x1:sx(P[0][0]),y1:sy(P[0][1]),x2:sx(P[1][0]),y2:sy(P[1][1]),stroke:col,'stroke-width':2.5});}"""


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


Rg = N["ring"]
ring = dict(x=Rg["x"], y=Rg["y"], label=Rg["label"])

wrap("I1", """<div class="wbox" id="w-i1">
<div class="wrow" style="align-items:flex-start;gap:14px">
<svg id="i1s" viewBox="0 0 380 380" style="max-width:380px"></svg>
<div style="min-width:240px"><div>准确率</div><div class="wn" id="i1v" style="font-size:2em"></div>
<div id="i1c" style="margin-top:10px"></div>
<div class="wrow"><label><input type="checkbox" id="i1one"> 只用第 1 条线</label></div>
<div class="wrow"><button id="i1g">看一种摆法</button></div></div></div>
<div class="cap">每条线是一个 logistic 单元：它把平面分成两边。蓝色阴影是“在每条线的内侧”的地方，机器把那里的点判为内圈（蓝点）。拖动滑块转动、平移三条线，让阴影正好盖住蓝点。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.getElementById('i1s');const box=document.getElementById('i1c');
const st=[{a:0,d:3.0},{a:25,d:3.0},{a:50,d:3.0}];const cols=['#00795A','#00795A','#00795A'];
st.forEach((s,k)=>{const r=document.createElement('div');r.style.marginTop='6px';
 r.innerHTML='第 '+(k+1)+' 条线　转动 <input type="range" min="0" max="359" step="1" value="'+s.a+'" data-k="'+k+'" data-f="a"><br>　　　　　平移 <input type="range" min="0.2" max="3.4" step="0.05" value="'+s.d+'" data-k="'+k+'" data-f="d">';box.appendChild(r);});
box.querySelectorAll('input').forEach(inp=>inp.addEventListener('input',()=>{st[+inp.dataset.k][inp.dataset.f]=+inp.value;draw();}));
const one=document.getElementById('i1one');one.addEventListener('change',draw);
function active(){return one.checked?st.slice(0,1):st;}
function inside(x,y){return active().every(s=>{const t=s.a*Math.PI/180;return x*Math.cos(t)+y*Math.sin(t)<s.d;});}
function draw(){svg.innerHTML='';shade(svg,inside);points(svg,D);
 active().forEach((s,k)=>{const t=s.a*Math.PI/180;line(svg,Math.cos(t),Math.sin(t),-s.d,cols[k]);});
 let ok=0;D.x.forEach((x,i)=>{if((inside(x,D.y[i])?1:0)===D.label[i])ok++;});
 document.getElementById('i1v').textContent=(ok/D.x.length).toFixed(3);}
document.getElementById('i1g').addEventListener('click',()=>{[20,140,260].forEach((a,k)=>{st[k].a=a;st[k].d=1.25;});one.checked=false;
 box.querySelectorAll('input').forEach(inp=>{inp.value=st[+inp.dataset.k][inp.dataset.f];});draw();});
draw();})();</script>""", ring)

I2 = """<div class="wbox" id="w-i2">
<div class="wrow"><span>隐藏单元个数：</span><button data-h="1">1</button><button data-h="2">2</button><button data-h="3">3</button><button data-h="4">4</button></div>
<div class="wrow"><span>中间那一步：</span><button data-a="sigmoid">sigmoid</button><button data-a="relu">另一种弯法</button><button data-a="linear">去掉</button></div>
<div class="wrow" style="align-items:flex-start;gap:14px"><svg id="i2s" viewBox="0 0 380 380" style="max-width:380px"></svg>
<div style="padding-top:20px;min-width:200px"><div>训练后的准确率</div><div class="wn" id="i2v" style="font-size:2em"></div><div id="i2m" style="margin-top:10px;color:#555;max-width:220px"></div></div></div>
<div class="cap">这里的线不是手摆的：每个网络都用交叉熵和梯度下降训练了 300 步，机器自己摆。绿线是每个隐藏单元的直线（它的输入加权和等于 0 的地方），蓝色阴影是网络判为内圈的地方。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.getElementById('i2s');let h=3,act='sigmoid';
const f={sigmoid:z=>1/(1+Math.exp(-z)),relu:z=>Math.max(0,z),linear:z=>z};
function prob(n,x,y){let s=n.b2;for(let k=0;k<n.b1.length;k++){s+=n.W2[k]*f[act](n.W1[0][k]*x+n.W1[1][k]*y+n.b1[k]);}return 1/(1+Math.exp(-s));}
function draw(){const n=D.nets[act+h];svg.innerHTML='';shade(svg,(x,y)=>prob(n,x,y)>0.5);points(svg,D);
 for(let k=0;k<h;k++)line(svg,n.W1[0][k],n.W1[1][k],n.b1[k],'#00795A');
 document.getElementById('i2v').textContent=n.acc.toFixed(3);
 document.getElementById('i2m').textContent=act==='linear'?'去掉中间那一步：不管几个单元，阴影的边界都只是一条直线。':'';
 document.querySelectorAll('#w-i2 button').forEach(b=>b.classList.toggle('on',!!((b.dataset.h&&+b.dataset.h===h)||(b.dataset.a&&b.dataset.a===act))));}
document.querySelectorAll('#w-i2 button[data-h]').forEach(b=>b.addEventListener('click',()=>{h=+b.dataset.h;draw();}));
document.querySelectorAll('#w-i2 button[data-a]').forEach(b=>b.addEventListener('click',()=>{act=b.dataset.a;draw();}));
draw();})();</script>"""


def i2(name, with_act):
    """I2 for Section 4 (activation buttons); I2a for Section 3 (sigmoid only, before the question about the middle)."""
    html = I2.replace("w-i2", "w-" + name).replace("i2s", name + "s").replace("i2v", name + "v").replace("i2m", name + "m")
    if with_act == "none":   # no buttons for the middle step at all
        html = re.sub(r'<div class="wrow"><span>中间那一步：</span>.*?</div>\n', "", html, flags=re.S)
    if with_act == "relu":  # slides, after ReLU has been named
        html = html.replace(">另一种弯法</button>", ">ReLU</button>")
    if with_act == "linear":  # sigmoid or none only (slides, before the second bend is offered)
        html = html.replace('<button data-a="relu">另一种弯法</button>', "")
    wrap(name, html, dict(ring, nets=Rg["nets"]))


i2("I2a", "none")
i2("I2", "full")
i2("I2n", "linear")
i2("I2r", "relu")
print("widgets ok")
