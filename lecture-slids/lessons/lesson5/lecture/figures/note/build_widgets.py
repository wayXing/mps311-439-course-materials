"""Lesson 5 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I1  try a straight line yourself on the hand-made data (the problem moment)
I2  place a point: its height under each class's ellipse, times the class share, gives the probability
I3  two candidates: one shared spread (LDA, straight) or one spread per class (QDA, curved)
I4  pull each spread towards a circle (reg_param): ellipses become round, the boundary straightens
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
.wbox input[type=range]{width:200px}
.wbox button{font:inherit;padding:3px 10px;border:2px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
</style>"""
# shared JS: drawing helpers, the plot frame, the two Gaussians
LIB = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}
const X0=-3.5,X1=5.5,Y0=-3.5,Y1=4.5,L=40,Rr=430,T=8,Bt=355;
const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
const ix=px=>X0+(px-L)/(Rr-L)*(X1-X0),iy=py=>Y0+(Bt-py)/(Bt-T)*(Y1-Y0);
function frame(svg){el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 [-2,0,2,4].forEach(v=>{el(svg,'text',{x:sx(v),y:Bt+16,'text-anchor':'middle','font-size':12,fill:'#444'},v);el(svg,'text',{x:L-7,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v);});
 el(svg,'text',{x:(L+Rr)/2,y:Bt+34,'text-anchor':'middle','font-size':13,fill:'#333'},'x₁');el(svg,'text',{x:12,y:(T+Bt)/2,'font-size':13,fill:'#333'},'x₂');}
function dots(svg,D,wrong){D.x1.forEach((x,k)=>{const y=D.x2[k],c=D.label[k];
 el(svg,'circle',c?{cx:sx(x),cy:sy(y),r:4.2,fill:'""" + B + """','fill-opacity':0.9}:{cx:sx(x),cy:sy(y),r:3.4,fill:'none',stroke:'#555','stroke-width':1.2});
 if(wrong&&wrong[k])el(svg,'circle',{cx:sx(x),cy:sy(y),r:7.2,fill:'none',stroke:'""" + R + """','stroke-width':1.3});});}
function inv2(S){const d=S[0][0]*S[1][1]-S[0][1]*S[1][0];return [[S[1][1]/d,-S[0][1]/d],[-S[1][0]/d,S[0][0]/d],d];}
function dens(x,y,mu,S){const [a,b,d]=inv2(S),u=x-mu[0],v=y-mu[1];const q=u*(a[0]*u+a[1]*v)+v*(b[0]*u+b[1]*v);return Math.exp(-0.5*q)/(2*Math.PI*Math.sqrt(d));}
function shrink(S,r){return [[(1-r)*S[0][0]+r,(1-r)*S[0][1]],[(1-r)*S[1][0],(1-r)*S[1][1]+r]];}
function post(x,y,D,S1,S0){const a=dens(x,y,D.mu['1'],S1)*D.n1,b=dens(x,y,D.mu['0'],S0)*D.n0;return a/(a+b);}
function ell(svg,mu,S,k,attr){const a=S[0][0],b=S[0][1],c=S[1][1],h=Math.sqrt((a-c)*(a-c)/4+b*b),l1=(a+c)/2+h,l2=(a+c)/2-h,th=0.5*Math.atan2(2*b,a-c);
 let d='';for(let i=0;i<=72;i++){const t=2*Math.PI*i/72,u=k*Math.sqrt(l1)*Math.cos(t),v=k*Math.sqrt(l2)*Math.sin(t);
 const x=mu[0]+u*Math.cos(th)-v*Math.sin(th),y=mu[1]+u*Math.sin(th)+v*Math.cos(th);d+=(i?'L':'M')+sx(x).toFixed(1)+' '+sy(y).toFixed(1);}
 const e=el(svg,'path',Object.assign({d:d,fill:'none'},attr));return e;}
function region(svg,f){const n=90,dx=(X1-X0)/n,dy=(Y1-Y0)/n,v=[];for(let i=0;i<=n;i++){v.push([]);for(let j=0;j<=n;j++)v[i].push(f(X0+i*dx,Y0+j*dy));}
 const g=el(svg,'g',{opacity:0.16});for(let i=0;i<n;i++)for(let j=0;j<n;j++){if((v[i][j]+v[i+1][j]+v[i][j+1]+v[i+1][j+1])/4>=0.5)
  el(g,'rect',{x:sx(X0+i*dx),y:sy(Y0+(j+1)*dy),width:(Rr-L)/n+.4,height:(Bt-T)/n+.4,fill:'""" + G + """'});}
 let d='';const cut=(x1,y1,a,x2,y2,b)=>{const t=(0.5-a)/(b-a);return [x1+t*(x2-x1),y1+t*(y2-y1)];};
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){const xa=X0+i*dx,xb=xa+dx,ya=Y0+j*dy,yb=ya+dy,a=v[i][j],b=v[i+1][j],c=v[i+1][j+1],e=v[i][j+1],p=[];
  if((a>=.5)!==(b>=.5))p.push(cut(xa,ya,a,xb,ya,b));if((b>=.5)!==(c>=.5))p.push(cut(xb,ya,b,xb,yb,c));
  if((e>=.5)!==(c>=.5))p.push(cut(xa,yb,e,xb,yb,c));if((a>=.5)!==(e>=.5))p.push(cut(xa,ya,a,xa,yb,e));
  for(let k=0;k+1<p.length;k+=2)d+='M'+sx(p[k][0]).toFixed(1)+' '+sy(p[k][1]).toFixed(1)+'L'+sx(p[k+1][0]).toFixed(1)+' '+sy(p[k+1][1]).toFixed(1);}
 return d;}
function edge(svg,d){el(svg,'path',{d:d,fill:'none',stroke:'""" + G + """','stroke-width':3,'stroke-linecap':'round'});}
function score(D,f){let ok=0;const wrong=D.x1.map((x,k)=>{const w=(f(x,D.x2[k])>=0.5)!==(D.label[k]===1);if(!w)ok++;return w;});return [ok/D.x1.length,wrong];}"""


def wrap(name, body, data):
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data)).replace("__LIB__", LIB) + "\n```\n"
    if EN:
        from i18n_en import tr, check
        html = check(tr(html), name)
    open(os.path.join(OUTDIR, f"{name}.html"), "w", encoding="utf8").write(html)
    # a translated string with an apostrophe can break a single-quoted JS string: check the syntax
    if shutil.which("node"):
        js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        res = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True)
        os.remove(fh.name)
        if res.returncode:
            raise SystemExit(f"JavaScript syntax error in {name}: {res.stderr[:300]}")


T = N["two"]
DATA = dict(x1=T["x1"], x2=T["x2"], label=T["label"], mu=T["mu"], cov=T["cov"], pooled=T["pooled"], n1=T["n1"], n0=T["n0"],
            lr=T["lr"], base=T["baseline"])

# ---------------- I1: try a straight line yourself ----------------
wrap("I1", """<div class="wbox" id="w-i1">
<svg viewBox="0 0 440 400" style="max-width:440px"></svg>
<div class="wrow"><label>方向 <input type="range" id="i1t" min="0" max="359" step="1" value="135"> <span class="wn" id="i1tv"></span></label>
<label>位置 <input type="range" id="i1c" min="-3" max="3" step="0.05" value="0"> <span class="wn" id="i1cv"></span></label></div>
<div class="wrow"><button id="i1lr">logistic regression 的直线</button>
<span>准确率 <span class="wn" id="i1a"></span></span><span>永远猜类 0：<span class="wn" id="i1base"></span></span></div>
<div class="cap">蓝点是类 1（120 个），灰色空心圆是类 0（240 个）。绿色一侧判为类 1，判错的点画橙圈。拖两个滑块，看能不能找到一条直线把两类分开；再点按钮，看 logistic regression 找到的直线。</div></div>
<script>(function(){
__LIB__
const D=__D__;const svg=document.querySelector('#w-i1 svg');const ts=document.getElementById('i1t'),cs=document.getElementById('i1c');
function draw(){const th=+ts.value*Math.PI/180,c=+cs.value,nx=Math.cos(th),ny=Math.sin(th);svg.innerHTML='';
 const f=(x,y)=>(nx*x+ny*y-c)>=0?1:0;region(svg,f);frame(svg);const [acc,wrong]=score(D,f);dots(svg,D,wrong);
 const cp=el(svg,'clipPath',{id:'i1clip'});el(cp,'rect',{x:L,y:T,width:Rr-L,height:Bt-T});
 const px=nx*c,py=ny*c;el(svg,'line',{x1:sx(px-9*ny),y1:sy(py+9*nx),x2:sx(px+9*ny),y2:sy(py-9*nx),stroke:'""" + G + """','stroke-width':3,'clip-path':'url(#i1clip)'});
 document.getElementById('i1tv').textContent=ts.value+'°';document.getElementById('i1cv').textContent=c.toFixed(2);document.getElementById('i1a').textContent=acc.toFixed(3);document.getElementById('i1base').textContent=D.base.toFixed(3);}
document.getElementById('i1lr').addEventListener('click',()=>{const w1=D.lr.w1,w2=D.lr.w2,n=Math.hypot(w1,w2);
 let deg=Math.atan2(w2,w1)*180/Math.PI;if(deg<0)deg+=360;ts.value=Math.round(deg);cs.value=(-D.lr.b/n).toFixed(2);draw();});
ts.addEventListener('input',draw);cs.addEventListener('input',draw);draw();})();</script>""", DATA)

# ---------------- I2: compare the heights, times the shares ----------------
wrap("I2", """<div class="wbox" id="w-i2">
<div class="wrow" style="align-items:flex-start;gap:10px">
<svg id="i2s" viewBox="0 0 440 400" style="max-width:440px;cursor:crosshair"></svg>
<svg id="i2b" viewBox="0 0 300 330" style="max-width:300px"></svg></div>
<div class="wrow" style="font-size:.88em;color:#444"><b>+</b> 是每一类的中心；每一类两层椭圆：实线内圈约含 40% 的点，虚线外圈约含 86%</div>
<div class="cap">每一类画成一个椭圆：十字是中心（平均值），内圈和外圈分别约含这一类 40% 和 86% 的点。点一下或拖动黑点，右边显示：黑点在每一类里有多常见（这一类的钟形“小山”在那里有多高），乘上这一类占的比例，两者谁大就判谁。</div></div>
<script>(function(){
__LIB__
const D=__D__;const svg=document.getElementById('i2s'),bar=document.getElementById('i2b');let P=[1.0,0.0],drag=false;const fmt=v=>v<0.001?'< 0.001':v.toFixed(3);
function draw(){svg.innerHTML='';frame(svg);dots(svg,D,null);
 [['1','""" + B + """'],['0','#555']].forEach(([c,col])=>{ell(svg,D.mu[c],D.cov[c],1,{stroke:col,'stroke-width':2.4});ell(svg,D.mu[c],D.cov[c],2,{stroke:col,'stroke-width':1.3,'stroke-dasharray':'5 4'});
  el(svg,'text',{x:sx(D.mu[c][0]),y:sy(D.mu[c][1])+9,'text-anchor':'middle','font-size':30,fill:col,'font-weight':700,stroke:'#fff','stroke-width':4,'paint-order':'stroke'},'+');});
 el(svg,'circle',{cx:sx(P[0]),cy:sy(P[1]),r:8,fill:'#111',stroke:'#fff','stroke-width':2});
 const h1=dens(P[0],P[1],D.mu['1'],D.cov['1']),h0=dens(P[0],P[1],D.mu['0'],D.cov['0']),s1=h1*D.n1/(D.n1+D.n0),s0=h0*D.n0/(D.n1+D.n0),p=s1/(s1+s0);
 bar.innerHTML='';const rows=[['在类 1 里有多常见',h1,'""" + B + """'],['在类 0 里有多常见',h0,'#777'],['× 比例 1/3',s1,'""" + B + """'],['× 比例 2/3',s0,'#777']];
 [[0,1],[2,3]].forEach((pair,g)=>{const m=Math.max(rows[pair[0]][1],rows[pair[1]][1],1e-12);pair.forEach((k,j)=>{const y0=14+g*120+j*48;
  el(bar,'text',{x:0,y:y0,'font-size':13,fill:'#333'},rows[k][0]);el(bar,'rect',{x:0,y:y0+6,width:Math.max(1,200*rows[k][1]/m),height:20,fill:rows[k][2],'fill-opacity':0.8});
  el(bar,'text',{x:206,y:y0+21,'font-size':13,fill:'#333'},fmt(rows[k][1]));});});
 el(bar,'text',{x:0,y:270,'font-size':14,fill:'#333'},'类 1 的概率');el(bar,'text',{x:0,y:300,'font-size':26,'font-weight':700,fill:p>=0.5?'""" + B + """':'#555'},p.toFixed(3));
 el(bar,'text',{x:110,y:300,'font-size':15,fill:'#333'},p>=0.5?'判为类 1':'判为类 0');}
function at(e){const r=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal;const px=(e.clientX-r.left)/r.width*vb.width,py=(e.clientY-r.top)/r.height*vb.height;
 P=[Math.max(X0,Math.min(X1,ix(px))),Math.max(Y0,Math.min(Y1,iy(py)))];draw();}
svg.addEventListener('pointerdown',e=>{drag=true;at(e);});svg.addEventListener('pointermove',e=>{if(drag)at(e);});window.addEventListener('pointerup',()=>{drag=false;});
draw();})();</script>""", DATA)

# ---------------- I3: one shared spread or one per class ----------------
wrap("I3", """<div class="wbox" id="w-i3">
<div class="wrow"><button data-k="lda">两类共用一个散布</button><button data-k="qda">每类有自己的散布</button>
<span>准确率 <span class="wn" id="i3a"></span></span></div>
<svg viewBox="0 0 440 400" style="max-width:440px"></svg>
<div class="wrow" style="font-size:.88em;color:#444">虚线椭圆：每一类的描述（约含 86% 的点）；绿色区域：判为类 1；<span style="color:#C84B00">◯</span> 判错</div>
<div class="cap">虚线椭圆是每一类的描述（约含 86% 的点）。“共用一个散布”时，两个椭圆形状、大小、方向都一样，只是中心不同；“各有散布”时，每类用自己的。绿色区域判为类 1，判错的点画橙圈。两个按钮来回切换，看边界的形状。</div></div>
<script>(function(){
__LIB__
const D=__D__;const svg=document.querySelector('#w-i3 svg');let k='lda';
function draw(){svg.innerHTML='';const S1=k==='lda'?D.pooled:D.cov['1'],S0=k==='lda'?D.pooled:D.cov['0'];const f=(x,y)=>post(x,y,D,S1,S0);
 const bd=region(svg,f);frame(svg);const [acc,wrong]=score(D,f);dots(svg,D,wrong);edge(svg,bd);
 ell(svg,D.mu['1'],S1,2,{stroke:'""" + B + """','stroke-width':2,'stroke-dasharray':'6 4'});ell(svg,D.mu['0'],S0,2,{stroke:'#555','stroke-width':2,'stroke-dasharray':'6 4'});
 document.getElementById('i3a').textContent=acc.toFixed(3);
 document.querySelectorAll('#w-i3 button').forEach(b=>{b.style.fontWeight=b.dataset.k===k?'700':'400';b.style.borderColor=b.dataset.k===k?'""" + G + """':'#ccc';});}
document.querySelectorAll('#w-i3 button').forEach(b=>b.addEventListener('click',()=>{k=b.dataset.k;draw();}));draw();})();</script>""", DATA)

# ---------------- I4: pull each spread towards a circle ----------------
wrap("I4", """<div class="wbox" id="w-i4">
<div class="wrow"><label>往“圆”收多少 r <input type="range" id="i4r" min="0" max="1" step="0.05" value="0"> <span class="wn" id="i4rv"></span></label>
<span>准确率 <span class="wn" id="i4a"></span></span></div>
<svg viewBox="0 0 440 400" style="max-width:440px"></svg>
<div class="wrow" style="font-size:.88em;color:#444">虚线椭圆：收缩后每一类的散布；绿色区域：判为类 1</div>
<div class="cap">每类自己的散布（r = 0，就是 QDA）和一个半径为 1 的圆按比例混合：散布 = (1 − r) × 自己的散布 + r × 圆。拖动 r，看两个虚线椭圆怎样变圆，绿色边界怎样变。</div></div>
<script>(function(){
__LIB__
const D=__D__;const svg=document.querySelector('#w-i4 svg'),rs=document.getElementById('i4r');
function draw(){const r=+rs.value;svg.innerHTML='';const S1=shrink(D.cov['1'],r),S0=shrink(D.cov['0'],r);const f=(x,y)=>post(x,y,D,S1,S0);
 const bd=region(svg,f);frame(svg);const [acc,wrong]=score(D,f);dots(svg,D,wrong);edge(svg,bd);
 ell(svg,D.mu['1'],S1,2,{stroke:'""" + B + """','stroke-width':2,'stroke-dasharray':'6 4'});ell(svg,D.mu['0'],S0,2,{stroke:'#555','stroke-width':2,'stroke-dasharray':'6 4'});
 document.getElementById('i4rv').textContent=r.toFixed(2);document.getElementById('i4a').textContent=acc.toFixed(3);}
rs.addEventListener('input',draw);draw();})();</script>""", DATA)
print("widgets ok")
