"""Lesson 6 Note: widgets I1x and I4 (no CDN), data from numbers_v2.json.
Run (from this folder): python build_widgets_v2.py
I1x  XOR: drag one cut, then switch on one more cut on each side
I4   regression tree, horsepower -> mpg: drag the depth, watch the staircase and two RMSEs
CSS and helper code are copied from ../note/build_widgets.py so the widgets look the same.
"""
import json, os, re, subprocess, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers_v2.json")))
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

EN = "en" in " ".join(os.sys.argv[1:]).split()   # python build_widgets_v2.py --lang en
TR = {  # Chinese -> English for the widget text
    "四团点，对角的两团是同一类。先拖第一刀：一条竖线把平面分成左右两块，每一块按多数判断。一刀能分开吗？再勾选“两边各再切一刀”，在左右两块里各拖一条横线。":
        "Four clusters; opposite clusters share a class. Drag the first cut: a vertical line splits the plane into two sides, and each side predicts its majority. Can one cut separate them? Then tick \"one more cut on each side\" and drag a horizontal line on each side.",
    "313 辆训练车：横轴马力，纵轴油耗（英里每加仑）。绿线是回归树的预测：每片叶子给出落进来的车的平均油耗，所以预测是一级一级的台阶。拖动深度，看台阶怎样变多、变碎，两个 RMSE 怎样变（单位：英里每加仑）。":
        "313 training cars: horsepower across, miles per gallon up. The green line is the regression tree: each leaf predicts the average mpg of its cars, so the prediction is a staircase. Drag the depth and watch the steps and the two RMSEs (in mpg).",
    "最好的一条斜线（logistic regression）：": "best straight line (logistic regression): ",
    "浅绿底色：这一块判为类 1": "light green: predicted class 1",
    "两边各再切一刀": "one more cut on each side",
    "5 折交叉验证 RMSE ": "5-fold CV RMSE ",
    "油耗（英里每加仑）": "miles per gallon",
    "第一刀：x₁ = ": "first cut: x₁ = ",
    "左边：x₂ = ": "left side: x₂ = ",
    "右边：x₂ = ": "right side: x₂ = ",
    "直线：": "straight line: ",
    "训练 RMSE ": "training RMSE ",
    "最大深度": "max depth",
    "叶子数 ": "leaves ",
    "准确率 ": "accuracy ",
    "判错": "wrong",
    "类 0": "class 0",
    "类 1": "class 1",
    "马力": "horsepower",
}

def wrap(name, body, data, extra=None):
    for k, v in (extra or {}).items():
        body = body.replace(k, v)
    if EN:  # English version for the English Note and the Slides; leftover Chinese stops the build
        for zh, en in sorted(TR.items(), key=lambda kv: -len(kv[0])):
            body = body.replace(zh, en)
        left = re.findall(r"[\u4e00-\u9fff]+", body)
        if left:
            raise SystemExit(f"untranslated text in {name}: {left[:5]}")
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data)).replace("__EL__", EL) + "\n```\n"
    out = os.path.join(HERE, "..", "note_en") if EN else HERE
    open(os.path.join(out, f"{name}.html"), "w", encoding="utf8").write(html)
    if shutil.which("node"):
        js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        res = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True)
        os.remove(fh.name)
        if res.returncode:
            raise SystemExit(f"JavaScript syntax error in {name}: {res.stderr[:300]}")

Xo = N["xor"]
wrap("I1x", """<div class="wbox" id="w-i1x">
<svg viewBox="0 0 420 420" style="max-width:420px"></svg>
<div class="wrow"><label>第一刀：x₁ = <input type="range" id="i1a" min="0" max="5" step="0.1" value="1.0"> <span class="wn" id="i1av"></span></label></div>
<div class="wrow"><label><input type="checkbox" id="i1on"> 两边各再切一刀</label></div>
<div class="wrow" id="i1more" style="display:none"><label>左边：x₂ = <input type="range" id="i1b" min="0" max="5" step="0.1" value="1.0"></label><label>右边：x₂ = <input type="range" id="i1c" min="0" max="5" step="0.1" value="4.0"></label></div>
<div class="wrow" style="font-size:.9em;color:#444"><span><span style="color:#0072B2">●</span> 类 1</span><span><span style="color:#777">○</span> 类 0</span><span><span style="color:#C84B00;font-size:1.2em">◯</span> 判错</span><span>浅绿底色：这一块判为类 1</span></div>
<div class="wrow"><span>准确率 <span class="wn" id="i1acc"></span></span><span>最好的一条斜线（logistic regression）：<span class="wn">__LOGIT__</span></span></div>
<div class="cap">四团点，对角的两团是同一类。先拖第一刀：一条竖线把平面分成左右两块，每一块按多数判断。一刀能分开吗？再勾选“两边各再切一刀”，在左右两块里各拖一条横线。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.querySelector('#w-i1x svg');const L=36,Rr=410,T=8,Bt=382,X0=-0.8,X1=5.8;
const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-X0)/(X1-X0)*(Bt-T);
const a_=document.getElementById('i1a'),b_=document.getElementById('i1b'),c_=document.getElementById('i1c'),on=document.getElementById('i1on');
function region(x,y,a,b,c,two){if(!two)return x<=a?0:1;return x<=a?(y<=b?0:1):(y<=c?2:3);}
function draw(){const a=+a_.value,b=+b_.value,c=+c_.value,two=on.checked;svg.innerHTML='';
 document.getElementById('i1more').style.display=two?'flex':'none';
 const n1=[0,0,0,0],n=[0,0,0,0];D.x.forEach((x,k)=>{const g=region(x,D.y[k],a,b,c,two);n[g]++;n1[g]+=D.label[k];});
 const say=n.map((m,g)=>m&&n1[g]*2>m?1:0);
 const boxes=two?[[X0,a,X0,b],[X0,a,b,X1],[a,X1,X0,c],[a,X1,c,X1]]:[[X0,a,X0,X1],[a,X1,X0,X1]];
 boxes.forEach((q,g)=>{if(say[g])el(svg,'rect',{x:sx(q[0]),y:sy(q[3]),width:sx(q[1])-sx(q[0]),height:sy(q[2])-sy(q[3]),fill:'""" + G + """','fill-opacity':0.15});});
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 [0,1,2,3,4,5].forEach(v=>{el(svg,'text',{x:sx(v),y:Bt+15,'text-anchor':'middle','font-size':12,fill:'#444'},v);el(svg,'text',{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v);});
 el(svg,'line',{x1:sx(a),x2:sx(a),y1:T,y2:Bt,stroke:'#333','stroke-width':2.4});
 if(two){el(svg,'line',{x1:L,x2:sx(a),y1:sy(b),y2:sy(b),stroke:'#333','stroke-width':2.4});el(svg,'line',{x1:sx(a),x2:Rr,y1:sy(c),y2:sy(c),stroke:'#333','stroke-width':2.4});}
 let ok=0;D.x.forEach((x,k)=>{const g=region(x,D.y[k],a,b,c,two),w=say[g]!==D.label[k];if(!w)ok++;dot(svg,sx(x),sy(D.y[k]),D.label[k],w);});
 document.getElementById('i1av').textContent=a.toFixed(1);document.getElementById('i1acc').textContent=(ok/D.x.length).toFixed(3);}
[a_,b_,c_].forEach(e=>e.addEventListener('input',draw));on.addEventListener('change',draw);draw();})();</script>""",
     dict(x=Xo["x"], y=Xo["y"], label=Xo["label"]), {"__LOGIT__": f"{Xo['logit']:.2f}"})

Rg = N["reg"]
wrap("I4", """<div class="wbox" id="w-i4">
<svg viewBox="0 0 520 380" style="max-width:520px"></svg>
<div class="wrow"><label>最大深度 <input type="range" id="i4d" min="1" max="10" step="1" value="1"> <span class="wn" id="i4dv"></span></label></div>
<div class="wrow"><span>叶子数 <span class="wn" id="i4n"></span></span><span style="color:""" + B + """">训练 RMSE <span class="wn" id="i4t"></span></span><span style="color:""" + G + """">5 折交叉验证 RMSE <span class="wn" id="i4c"></span></span><span style="color:#555">直线：__LIN__</span></div>
<div class="cap">313 辆训练车：横轴马力，纵轴油耗（英里每加仑）。绿线是回归树的预测：每片叶子给出落进来的车的平均油耗，所以预测是一级一级的台阶。拖动深度，看台阶怎样变多、变碎，两个 RMSE 怎样变（单位：英里每加仑）。</div></div>
<script>(function(){
__EL__
const D=__D__;const svg=document.querySelector('#w-i4 svg'),sl=document.getElementById('i4d');
const L=46,Rr=510,T=8,Bt=332,X0=40,X1=235,Y0=5,Y1=50;const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
function draw(){const dp=String(sl.value),q=D.depths[dp];svg.innerHTML='';
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 [50,100,150,200].forEach(v=>el(svg,'text',{x:sx(v),y:Bt+15,'text-anchor':'middle','font-size':12,fill:'#444'},v));[10,20,30,40,50].forEach(v=>el(svg,'text',{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'},v));
 el(svg,'text',{x:(L+Rr)/2,y:Bt+34,'text-anchor':'middle','font-size':13,fill:'#333'},'马力');el(svg,'text',{x:12,y:(T+Bt)/2,'font-size':13,fill:'#333',transform:'rotate(-90 12 '+(T+Bt)/2+')','text-anchor':'middle'},'油耗（英里每加仑）');
 D.x.forEach((x,k)=>el(svg,'circle',{cx:sx(x),cy:sy(D.y[k]),r:3,fill:'""" + B + """','fill-opacity':0.45}));
 let p='';D.grid.forEach((x,i)=>{const yy=sy(q.curve[i]);p+=(i?'L'+sx(x).toFixed(1)+' '+sy(q.curve[i-1]).toFixed(1)+'L':'M')+sx(x).toFixed(1)+' '+yy.toFixed(1);});
 el(svg,'path',{d:p,fill:'none',stroke:'""" + G + """','stroke-width':2.6});
 document.getElementById('i4dv').textContent=dp;document.getElementById('i4n').textContent=q.leaves;
 document.getElementById('i4t').textContent=q.train.toFixed(2);document.getElementById('i4c').textContent=q.cv.toFixed(2);}
sl.addEventListener('input',draw);draw();})();</script>""",
     dict(x=Rg["x"], y=Rg["y"], grid=Rg["grid"], depths={k: v for k, v in Rg["depths"].items() if k != "None"}), {"__LIN__": f"{Rg['linear_cv']:.2f}"})
print("widgets v2 ok")
