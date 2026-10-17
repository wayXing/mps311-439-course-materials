"""Lesson 4 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I2  three sliders (w1, w2, b) on two standardised features: the line p = 0.5 turns and shifts.
I1  try a straight line on 0/1 labels yourself; I3  compare three squashing functions;
I6  the threshold as a cut sliding along the S curve, with the misclassified lumps circled.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
EN = "--lang" in os.sys.argv and os.sys.argv[os.sys.argv.index("--lang") + 1] == "en"
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
.wbox table{border-collapse:collapse;margin-top:6px;width:auto !important;display:inline-table}
.wbox td{border:1px solid #d0d7de;padding:3px 10px;text-align:center}
</style>"""
EL = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}"""


def wrap(name, body, data):
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data)) + "\n```\n"
    if EN:
        from i18n_en import tr, check
        html = check(tr(html), name)
    open(os.path.join(OUTDIR, f"{name}.html"), "w", encoding="utf8").write(html)
    # a translated string with an apostrophe can break a single-quoted JS string: check the syntax
    import re, subprocess, tempfile, shutil
    if shutil.which("node"):
        js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        res = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True)
        os.remove(fh.name)
        if res.returncode:
            raise SystemExit(f"JavaScript syntax error in {name}: {res.stderr[:300]}")


# ---------------- I2: the decision boundary of two features ----------------
T2 = N["two"]
wrap("I2", """<div class="wbox" id="w-i2">
<svg viewBox="0 0 460 400" style="max-width:460px"></svg>
<div class="wrow"><label>w₁（半径）<input type="range" id="i2a" min="-4" max="4" step="0.1" value="1"> <span class="wn" id="i2av"></span></label>
<label>w₂（纹理）<input type="range" id="i2b" min="-4" max="4" step="0.1" value="0"> <span class="wn" id="i2bv"></span></label>
<label>b <input type="range" id="i2c" min="-3" max="3" step="0.1" value="0"> <span class="wn" id="i2cv"></span></label></div>
<div class="wrow"><span>按 p = 0.5 判断的准确率 = <span class="wn" id="i2r"></span></span></div>
<div class="cap">蓝点是恶性，灰色空心圆是良性，两个特征都已标准化。绿线是 p = 0.5 的位置，绿色越深，模型给“恶性”的概率越高。先只动 w₁ 和 w₂，再只动 b，看线怎样变。</div></div>
<script>(function(){
""" + EL + """
const D=__D__;const svg=document.querySelector('#w-i2 svg');
const L=46,Rr=450,T=8,Bt=360,X0=-2.5,X1=4.5,Y0=-2.5,Y1=4.5;
const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
const sl=['i2a','i2b','i2c'].map(i=>document.getElementById(i));
function draw(){const w1=+sl[0].value,w2=+sl[1].value,b=+sl[2].value;svg.innerHTML='';
 const n=28,dx=(X1-X0)/n,dy=(Y1-Y0)/n;
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){const x=X0+(i+.5)*dx,y=Y0+(j+.5)*dy,p=1/(1+Math.exp(-(w1*x+w2*y+b)));
  el(svg,'rect',{x:sx(X0+i*dx),y:sy(Y0+(j+1)*dy),width:(Rr-L)/n+.5,height:(Bt-T)/n+.5,fill:'""" + G + """','fill-opacity':(0.32*p).toFixed(3)});}
 for(let v=-2;v<=4;v+=2){el(svg,'text',{x:sx(v),y:Bt+16,'text-anchor':'middle','font-size':12,fill:'#444'},v);el(svg,'text',{x:L-8,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v);}
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 el(svg,'text',{x:(L+Rr)/2,y:392,'text-anchor':'middle','font-size':13,fill:'#333'},'细胞核平均半径（标准化）');
 el(svg,'text',{x:14,y:(T+Bt)/2,'text-anchor':'middle','font-size':13,fill:'#333',transform:'rotate(-90 14 '+(T+Bt)/2+')'},'细胞核纹理（标准化）');
 let ce=0,ok=0;
 D.z1.forEach((x,k)=>{const y=D.z2[k],c=D.label[k],p=1/(1+Math.exp(-(w1*x+w2*y+b)));
  ce+=-(c?Math.log(Math.max(p,1e-12)):Math.log(Math.max(1-p,1e-12)));ok+=((p>=0.5)===(c===1))?1:0;
  if(x>X0&&x<X1&&y>Y0&&y<Y1)el(svg,'circle',c?{cx:sx(x),cy:sy(y),r:3.2,fill:'""" + B + """','fill-opacity':0.75}:{cx:sx(x),cy:sy(y),r:3.2,fill:'none',stroke:'#777'});});
 if(Math.abs(w2)>1e-9){const ya=-(w1*X0+b)/w2,yb=-(w1*X1+b)/w2;el(svg,'line',{x1:sx(X0),y1:sy(ya),x2:sx(X1),y2:sy(yb),stroke:'""" + G + """','stroke-width':2.6,'clip-path':'url(#i2clip)'});}
 else if(Math.abs(w1)>1e-9){const xv=-b/w1;el(svg,'line',{x1:sx(xv),y1:sy(Y0),x2:sx(xv),y2:sy(Y1),stroke:'""" + G + """','stroke-width':2.6,'clip-path':'url(#i2clip)'});}
 const cp=el(svg,'clipPath',{id:'i2clip'});el(cp,'rect',{x:L,y:T,width:Rr-L,height:Bt-T});
 document.getElementById('i2av').textContent=w1.toFixed(1);document.getElementById('i2bv').textContent=w2.toFixed(1);document.getElementById('i2cv').textContent=b.toFixed(1);
 document.getElementById('i2r').textContent=(ok/D.z1.length).toFixed(3);}
sl.forEach(s=>s.addEventListener('input',draw));draw();})();</script>""", dict(z1=T2["z1"], z2=T2["z2"], label=T2["label"]))

# ---------------- I1: try a straight line yourself ----------------
O = N["one"]
L1 = N["line"]
wrap("I1", """<div class="wbox" id="w-i1">
<svg viewBox="0 0 600 380" style="max-width:600px"></svg>
<div class="wrow"><label>斜率 w <input type="range" id="i1w" min="-0.05" max="0.25" step="0.002" value="0.05"> <span class="wn" id="i1wv"></span></label>
<label>截距 b <input type="range" id="i1b" min="-3" max="2" step="0.01" value="-0.3"> <span class="wn" id="i1bv"></span></label></div>
<div class="wrow"><button id="i1ls">最小二乘直线</button><label><input type="checkbox" id="i1far"> 加入 20 个半径 40 的假想肿块（都是恶性）</label></div>
<div class="wrow"><span>预测落在 0 到 1 之外：<span class="wn" id="i1out"></span></span><span>按 0.5 切开的位置：半径 <span class="wn" id="i1cut"></span></span></div>
<div class="cap">每个点是一个训练肿块：上排蓝点是恶性（1），下排灰圈是良性（0），上下稍微错开以免重叠。先自己拖两个滑块，看能不能让直线既贴近两排点、又不跑出 0 到 1（红色区域）；再点“最小二乘直线”；最后勾选假想肿块，看最小二乘直线和切开的位置怎样变。</div></div>
<script>(function(){
""" + EL + """
const D=__D__;const svg=document.querySelector('#w-i1 svg');
const L=50,Rr=590,T=10,Bt=330,X0=5,X1=42,Y0=-0.7,Y1=1.9;
const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
const ws=document.getElementById('i1w'),bs=document.getElementById('i1b'),far=document.getElementById('i1far');
const jit=D.r.map((v,k)=>((k*7919)%97)/97*0.16-0.08);
function data(){const xs=D.r.slice(),ys=D.y.slice();if(far.checked){for(let k=0;k<20;k++){xs.push(40);ys.push(1);}}return [xs,ys];}
let W=+ws.value,Bb=+bs.value;
function draw(){const w=W,b=Bb;svg.innerHTML='';const [xs,ys]=data();
 el(svg,'rect',{x:L,y:sy(Y1),width:Rr-L,height:sy(1)-sy(Y1),fill:'""" + R + """','fill-opacity':0.1});
 el(svg,'rect',{x:L,y:sy(0),width:Rr-L,height:sy(Y0)-sy(0),fill:'""" + R + """','fill-opacity':0.1});
 for(let v=10;v<=40;v+=10){el(svg,'text',{x:sx(v),y:Bt+18,'text-anchor':'middle','font-size':13,fill:'#444'},v);}
 [0,0.5,1].forEach(v=>el(svg,'text',{x:L-8,y:sy(v)+4,'text-anchor':'end','font-size':13,fill:'#444'},v));
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 el(svg,'text',{x:(L+Rr)/2,y:370,'text-anchor':'middle','font-size':14,fill:'#333'},'细胞核平均半径');
 xs.forEach((x,k)=>{const yy=ys[k]+(k<jit.length?jit[k]:((k*13)%7)/7*0.16-0.08);
  el(svg,'circle',ys[k]?{cx:sx(x),cy:sy(yy),r:4.5,fill:'""" + B + """','fill-opacity':0.75}:{cx:sx(x),cy:sy(yy),r:4.5,fill:'none',stroke:'#777','stroke-width':1.2});});
 const cp=el(svg,'clipPath',{id:'i1clip'});el(cp,'rect',{x:L,y:T,width:Rr-L,height:Bt-T});
 el(svg,'line',{x1:sx(X0),y1:sy(w*X0+b),x2:sx(X1),y2:sy(w*X1+b),stroke:'""" + G + """','stroke-width':3,'clip-path':'url(#i1clip)'});
 let out=0;xs.forEach(x=>{const v=w*x+b;if(v<0||v>1)out++;});
 document.getElementById('i1out').textContent=Math.round(100*out/xs.length)+'%';
 if(w>1e-6){const c=(0.5-b)/w;if(c>X0&&c<X1){el(svg,'line',{x1:sx(c),x2:sx(c),y1:T,y2:Bt,stroke:'""" + G + """','stroke-dasharray':'6 4','stroke-width':1.6});}
  document.getElementById('i1cut').textContent=c.toFixed(2);}else{document.getElementById('i1cut').textContent='—';}
 document.getElementById('i1wv').textContent=w.toFixed(3);document.getElementById('i1bv').textContent=b.toFixed(2);}
function ls(){const [xs,ys]=data();const n=xs.length,mx=xs.reduce((a,c)=>a+c,0)/n,my=ys.reduce((a,c)=>a+c,0)/n;
 let sxy=0,sxx=0;xs.forEach((x,k)=>{sxy+=(x-mx)*(ys[k]-my);sxx+=(x-mx)*(x-mx);});const w=sxy/sxx;W=w;Bb=my-w*mx;ws.value=W;bs.value=Bb;draw();}
ws.addEventListener('input',()=>{W=+ws.value;draw();});bs.addEventListener('input',()=>{Bb=+bs.value;draw();});far.addEventListener('change',draw);
document.getElementById('i1ls').addEventListener('click',ls);draw();})();</script>""", dict(r=O["r"], y=O["y"]))

# ---------------- I3: look for a squashing function ----------------
wrap("I3", """<div class="wbox" id="w-i3">
<div class="wrow"><span>压缩方式：</span><button data-k="clip">截断到 0 到 1</button><button data-k="step">阶跃（大于 0 就是 1）</button><button data-k="sig">S 形：1 / (1 + e⁻ᶻ)</button></div>
<div class="wrow" style="align-items:flex-start;gap:10px">
<svg id="i3g" viewBox="0 0 260 230" style="max-width:260px"></svg>
<svg id="i3d" viewBox="0 0 470 300" style="max-width:470px"></svg></div>
<div class="wrow"><b id="i3txt"></b></div>
<div class="cap">左：压缩函数本身，横轴是直线的输出 z（可以是任何数），纵轴是压缩后的输出。右：用它压缩后，模型在训练肿块上画出的曲线。截断用的是最小二乘直线；阶跃和 S 形用的是同一条直线 z = wx + b。三个都试一遍：哪一个既在 0 到 1 之间，又说得出“有多确定”？</div></div>
<script>(function(){
""" + EL + """
const D=__D__;const g=document.getElementById('i3g'),dv=document.getElementById('i3d');
const msg={clip:'输出终于在 0 到 1 之间，但有硬折角，不平滑；中间那条直线仍靠最小二乘去拟合，远处的点照样拉动它。',
 step:'只剩 0 和 1：说不出“有多确定”；参数稍微改一点，输出要么不变、要么突然跳，没法一步步调。',
 sig:'处处在 0 到 1 之间；分界处是 0.5（拿不准），离得越远越确定；处处平滑，参数可以一点点调。'};
const fn={clip:z=>Math.max(0,Math.min(1,z)),step:z=>z>0?1:0,sig:z=>1/(1+Math.exp(-z))};
let k='clip';
function axes(svg,L,Rr,T,Bt,xl,xr,xt,lab){el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 xt.forEach(v=>el(svg,'text',{x:L+(v-xl)/(xr-xl)*(Rr-L),y:Bt+16,'text-anchor':'middle','font-size':12,fill:'#444'},v));
 [0,0.5,1].forEach(v=>el(svg,'text',{x:L-6,y:Bt-v*(Bt-T)/1.0*0.8-(Bt-T)*0.1+4,'text-anchor':'end','font-size':12,fill:'#444'},v));
 el(svg,'text',{x:(L+Rr)/2,y:Bt+34,'text-anchor':'middle','font-size':13,fill:'#333'},lab);}
function draw(){g.innerHTML='';dv.innerHTML='';
 const L=34,Rr=250,T=8,Bt=190,syg=v=>Bt-(Bt-T)*0.1-v*(Bt-T)*0.8,sxg=z=>L+(z+6)/12*(Rr-L);
 axes(g,L,Rr,T,Bt,-6,6,[-6,-3,0,3,6],'直线的输出 z');
 let d='';for(let i=0;i<=240;i++){const z=-6+12*i/240;d+=(i?'L':'M')+sxg(z).toFixed(1)+' '+syg(fn[k](z)).toFixed(1);}
 el(g,'path',{d:d,fill:'none',stroke:'""" + G + """','stroke-width':3});
 const L2=40,R2=460,T2=8,B2=250,X0=5,X1=30,sx=x=>L2+(x-X0)/(X1-X0)*(R2-L2),sy=v=>B2-(B2-T2)*0.1-v*(B2-T2)*0.8;
 axes(dv,L2,R2,T2,B2,X0,X1,[10,15,20,25,30],'细胞核平均半径');
 D.r.forEach((x,i)=>{if(x>X1)return;const yy=D.y[i]+(((i*7919)%97)/97*0.12-0.06);
  el(dv,'circle',D.y[i]?{cx:sx(x),cy:sy(yy),r:3.5,fill:'""" + B + """','fill-opacity':0.7}:{cx:sx(x),cy:sy(yy),r:3.5,fill:'none',stroke:'#777'});});
 let d2='';for(let i=0;i<=300;i++){const x=X0+(X1-X0)*i/300;const z=k==='clip'?D.lw*x+D.lb:D.gw*x+D.gb;d2+=(i?'L':'M')+sx(x).toFixed(1)+' '+sy(fn[k](z)).toFixed(1);}
 el(dv,'path',{d:d2,fill:'none',stroke:'""" + G + """','stroke-width':3});
 document.getElementById('i3txt').textContent=msg[k];
 document.querySelectorAll('#w-i3 button').forEach(b=>{b.style.fontWeight=b.dataset.k===k?'700':'400';b.style.borderColor=b.dataset.k===k?'""" + G + """':'#ccc';});}
document.querySelectorAll('#w-i3 button').forEach(b=>b.addEventListener('click',()=>{k=b.dataset.k;draw();}));draw();})();</script>""",
     dict(r=O["r"], y=O["y"], lw=L1["w"], lb=L1["b"], gw=N["logit1"]["w"], gb=N["logit1"]["b"]))

# ---------------- I6: the threshold is a cut that slides along the S curve ----------------
H = N["hist"]
wrap("I6", """<div class="wbox" id="w-i6">
<svg id="i6s" viewBox="0 0 640 360" style="max-width:640px"></svg>
<div class="wrow"><label>阈值 t <input type="range" id="i6t" min="0.02" max="0.98" step="0.01" value="0.5"> <span class="wn" id="i6tv"></span></label></div>
<div class="wrow" style="font-size:1.15em"><span>漏诊：<span class="wn" id="i6fn" style="color:""" + R + """"></span></span><span>误报：<span class="wn" id="i6fp" style="color:""" + R + """"></span></span><span>召回率 <span class="wn" id="i6rc"></span></span><span>精确率 <span class="wn" id="i6pr"></span></span></div>
<div class="cap">模型用全部 30 个测量值。426 个训练肿块，每个的概率来自 5 折交叉验证（它所在的那一折没有参加训练）。横轴是模型算出的 z，绿色 S 形曲线把它压成概率 p。上排蓝点是恶性，下排灰圈是良性。阈值 t 是曲线上的一个高度，对应横轴上的一条竖线：竖线右边判为恶性。拖动阈值，竖线左右移动；被判错的点画上橙圈。z 小于 −10 或大于 10 的点画在两端，所以两端各堆成一条竖线。</div></div>
<script>(function(){
""" + EL + """
const D=__D__;const svg=document.getElementById('i6s'),sl=document.getElementById('i6t');
const L=50,Rr=600,T=44,Bt=300,Z0=-10,Z1=10,sx=z=>L+(Math.max(Z0,Math.min(Z1,z))-Z0)/(Z1-Z0)*(Rr-L),sy=v=>Bt-(Bt-T)*0.08-v*(Bt-T)*0.84;
const lg=p=>{const q=Math.min(Math.max(p,1e-5),1-1e-5);return Math.log(q/(1-q));};
const pts=[];D.mal.forEach((p,i)=>pts.push([lg(p),1,((i*7919)%97)/97*0.1-0.05]));D.ben.forEach((p,i)=>pts.push([lg(p),0,((i*104729)%89)/89*0.1-0.05]));
function draw(){const t=+sl.value,zc=lg(t);svg.innerHTML='';
 el(svg,'rect',{x:sx(zc),y:T,width:Rr-sx(zc),height:Bt-T,fill:'""" + G + """','fill-opacity':0.08});
 el(svg,'text',{x:sx(zc)+8,y:sy(0.2),'font-size':14,fill:'""" + G + """','font-weight':700},'判为恶性 →');
 el(svg,'text',{x:sx(zc)-8,y:sy(0.8),'text-anchor':'end','font-size':14,fill:'#555','font-weight':700},'← 判为良性');
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 [[-10,'≤ −10'],[-5,'−5'],[0,'0'],[5,'5'],[10,'≥ 10']].forEach(([v,lab])=>el(svg,'text',{x:sx(v),y:Bt+18,'text-anchor':'middle','font-size':13,fill:'#444'},lab));
 el(svg,'text',{x:14,y:(T+Bt)/2,'text-anchor':'middle','font-size':13,fill:'#333',transform:'rotate(-90 14 '+((T+Bt)/2)+')'},'恶性的概率 p');
 el(svg,'circle',{cx:L+18,cy:14,r:4.5,fill:'""" + B + """','fill-opacity':0.75});el(svg,'text',{x:L+28,y:18,'font-size':12,fill:'#333'},'恶性');
 el(svg,'circle',{cx:L+128,cy:14,r:4.5,fill:'none',stroke:'#777','stroke-width':1.2});el(svg,'text',{x:L+138,y:18,'font-size':12,fill:'#333'},'良性');
 el(svg,'circle',{cx:L+228,cy:14,r:8,fill:'none',stroke:'""" + R + """','stroke-width':2.2});el(svg,'text',{x:L+242,y:18,'font-size':12,fill:'#333'},'判错');
 [0,0.5,1].forEach(v=>el(svg,'text',{x:L-8,y:sy(v)+4,'text-anchor':'end','font-size':13,fill:'#444'},v));
 el(svg,'text',{x:(L+Rr)/2,y:340,'text-anchor':'middle','font-size':14,fill:'#333'},'模型算出的 z（右边越像恶性）');
 let d='';for(let i=0;i<=200;i++){const z=Z0+(Z1-Z0)*i/200;d+=(i?'L':'M')+sx(z).toFixed(1)+' '+sy(1/(1+Math.exp(-z))).toFixed(1);}
 el(svg,'path',{d:d,fill:'none',stroke:'""" + G + """','stroke-width':3});
 el(svg,'line',{x1:L,x2:sx(zc),y1:sy(t),y2:sy(t),stroke:'#333','stroke-dasharray':'5 4'});
 el(svg,'line',{x1:sx(zc),x2:sx(zc),y1:T,y2:Bt,stroke:'#333','stroke-width':2.2});
 el(svg,'text',{x:L+6,y:sy(t)-6,'font-size':13,fill:'#333'},'t = '+t.toFixed(2));
 let fn=0,fp=0,tp=0;
 pts.forEach(([z,c,j])=>{const right=z>=zc,wrong=c?!right:right;if(c&&right)tp++;if(c&&!right)fn++;if(!c&&right)fp++;
  const cy=sy(c+j);el(svg,'circle',c?{cx:sx(z),cy:cy,r:4.5,fill:'""" + B + """','fill-opacity':0.75}:{cx:sx(z),cy:cy,r:4.5,fill:'none',stroke:'#777','stroke-width':1.2});
  if(wrong)el(svg,'circle',{cx:sx(z),cy:cy,r:8,fill:'none',stroke:'""" + R + """','stroke-width':2.2});});
 document.getElementById('i6tv').textContent=t.toFixed(2);document.getElementById('i6fn').textContent=fn;document.getElementById('i6fp').textContent=fp;
 document.getElementById('i6rc').textContent=(tp/(tp+fn)).toFixed(3);document.getElementById('i6pr').textContent=tp+fp?(tp/(tp+fp)).toFixed(3):'—';}
sl.addEventListener('input',draw);draw();})();</script>""", dict(mal=H["mal"], ben=H["ben"]))
print("widgets ok")
