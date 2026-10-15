"""Lesson 3 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I3  degree slider on the ten cars;  I5  lambda slider, degree 7, same ten cars and same axes.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
EN = "--lang" in os.sys.argv and os.sys.argv[os.sys.argv.index("--lang") + 1] == "en"
OUTDIR = os.path.join(HERE, "..", "note_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
N = json.load(open(os.path.join(HERE, "numbers.json")))
W = N["widget"]
B, G, R = "#0072B2", "#00795A", "#C84B00"
CSS = """<style>
.wbox{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:1.2em 0;background:#fafbfc;font-size:.95em}
.wbox svg{display:block;max-width:100%;height:auto}
.wbox .wrow{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin-top:8px}
.wbox .wn{font-variant-numeric:tabular-nums;font-weight:700}
.wbox .cap{color:#555;font-size:.9em;margin-top:6px}
.wbox input[type=range]{width:240px}
</style>"""

# shared plotting code: axes, test cars, ten cars, one curve
PLOT = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}
const L=50,Rr=540,T=12,Bt=262,X0=40,X1=235,Y0=0,Y1=50;
const sx=x=>L+(x-X0)/(X1-X0)*(Rr-L),sy=y=>Bt-(y-Y0)/(Y1-Y0)*(Bt-T);
function base(svg,D){svg.innerHTML='';
 const cp=el(svg,'clipPath',{id:svg.dataset.cid});el(cp,'rect',{x:L,y:T,width:Rr-L,height:Bt-T});
 for(let x=50;x<=230;x+=30){el(svg,'line',{x1:sx(x),x2:sx(x),y1:Bt,y2:Bt+4,stroke:'#888'});el(svg,'text',{x:sx(x),y:Bt+17,'text-anchor':'middle','font-size':12,fill:'#444'},x);}
 for(let y=0;y<=50;y+=10){el(svg,'line',{x1:L-4,x2:L,y1:sy(y),y2:sy(y),stroke:'#888'});el(svg,'text',{x:L-8,y:sy(y)+4,'text-anchor':'end','font-size':12,fill:'#444'},y);}
 el(svg,'line',{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'});el(svg,'line',{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'});
 el(svg,'text',{x:(L+Rr)/2,y:298,'text-anchor':'middle','font-size':13,fill:'#333'},'马力 hp');
 el(svg,'text',{x:14,y:(T+Bt)/2,'text-anchor':'middle','font-size':13,fill:'#333',transform:'rotate(-90 14 '+(T+Bt)/2+')'},'mpg');
 D.test_hp.forEach((x,i)=>el(svg,'circle',{cx:sx(x),cy:sy(D.test_mpg[i]),r:3.2,fill:'none',stroke:'#999','stroke-width':1}));}
function curve(svg,D,ys){let d='';ys.forEach((v,i)=>{const yy=Math.max(-200,Math.min(Y1+200,v));d+=(i?'L':'M')+sx(D.grid[i]).toFixed(1)+' '+sy(yy).toFixed(1);});
 el(svg,'path',{d:d,fill:'none',stroke:'""" + G + """','stroke-width':2.6,'clip-path':'url(#'+svg.dataset.cid+')'});}
function cars(svg,D){D.hp.forEach((x,i)=>el(svg,'circle',{cx:sx(x),cy:sy(D.mpg[i]),r:6.5,fill:'""" + B + """'}));}
function fmt(v){return v>=100?'大于 100':v.toFixed(2);}
function bars(svg,co){svg.innerHTML='';const L2=50,R2=540,T2=8,B2=112,top=7;
 const sy2=v=>B2-Math.max(0,Math.min(top,v))/top*(B2-T2);
 for(let k=0;k<=top;k++){el(svg,'line',{x1:L2-4,x2:R2,y1:sy2(k),y2:sy2(k),stroke:k?'#eee':'#888'});
  el(svg,'text',{x:L2-8,y:sy2(k)+4,'text-anchor':'end','font-size':11,fill:'#444'},k===0?'1':'10'+['','','²','³','⁴','⁵','⁶','⁷'][k]);}
 const n=7,bw=(R2-L2)/n;
 co.forEach((v,j)=>{const h=Math.log10(Math.max(Math.abs(v),1));const x=L2+j*bw+bw*0.2;
  el(svg,'rect',{x:x,y:sy2(h),width:bw*0.6,height:B2-sy2(h),fill:v>=0?'""" + G + """':'#fff',stroke:'""" + G + """','stroke-width':2});
  el(svg,'text',{x:x+bw*0.3,y:B2+15,'text-anchor':'middle','font-size':11,fill:'#444'},'w'+(j+1));});
 el(svg,'text',{x:14,y:(T2+B2)/2,'text-anchor':'middle','font-size':12,fill:'#333',transform:'rotate(-90 14 '+(T2+B2)/2+')'},'|系数|');}"""


def wrap(name, body, data):
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data)) + "\n```\n"
    if EN:
        from i18n_en import tr, check
        html = check(tr(html), name)
    open(os.path.join(OUTDIR, f"{name}.html"), "w", encoding="utf8").write(html)


common = dict(grid=W["grid"], test_hp=W["test_hp"], test_mpg=W["test_mpg"], hp=N["ten"]["hp"], mpg=N["ten"]["mpg"])

# ---------------- I3: degree ----------------
d3 = dict(common, curves=[W["deg_curves"][str(d)] for d in range(1, 8)], rows=N["ten"]["degrees"])
wrap("I3", """<div class="wbox" id="w-i3">
<svg viewBox="0 0 560 306" data-cid="c-i3" style="max-width:560px"></svg>
<svg id="i3c" viewBox="0 0 560 130" style="max-width:560px"></svg>
<div class="wrow"><label>阶数 <input type="range" id="i3d" min="1" max="7" step="1" value="1"> <span class="wn" id="i3dv"></span></label></div>
<div class="wrow"><span>训练 RMSE（10 辆车）= <span class="wn" id="i3a"></span></span><span>测试 RMSE（79 辆新车）= <span class="wn" id="i3b"></span></span></div>
<div class="cap">蓝点是用来拟合的 10 辆车，灰色空心圆是 79 辆测试车，绿色曲线是拟合结果。下方是各个系数（特征已标准化）的大小，纵轴是对数刻度，每格相差 10 倍；实心为正，空心为负。先预测，再把阶数一格一格往上拖。</div></div>
<script>(function(){
""" + PLOT + """
const D=__D__;const svg=document.querySelector('#w-i3 svg'),sl=document.getElementById('i3d');
function draw(){const k=+sl.value-1;base(svg,D);curve(svg,D,D.curves[k]);cars(svg,D);bars(document.getElementById('i3c'),D.rows[k].coefs);
 document.getElementById('i3dv').textContent=k+1;document.getElementById('i3a').textContent=fmt(D.rows[k].train);document.getElementById('i3b').textContent=fmt(D.rows[k].test);}
sl.addEventListener('input',draw);draw();})();</script>""", d3)

# ---------------- I5: lambda ----------------
d5 = dict(common, lams=W["lams"], curves=W["ridge_curves"], rows=W["ridge_rows"])
n5 = len(W["lams"]) - 1
wrap("I5", f"""<div class="wbox" id="w-i5">
<svg viewBox="0 0 560 306" data-cid="c-i5" style="max-width:560px"></svg>
<svg id="i5c" viewBox="0 0 560 130" style="max-width:560px"></svg>
<div class="wrow"><label>λ <input type="range" id="i5l" min="0" max="{n5}" step="1" value="0"> <span class="wn" id="i5lv"></span></label></div>
<div class="wrow"><span>系数的大小 ‖w‖ = <span class="wn" id="i5n"></span></span><span>训练 RMSE = <span class="wn" id="i5a"></span></span><span>测试 RMSE = <span class="wn" id="i5b"></span></span></div>
<div class="cap">同样的 10 辆车，阶数固定为 7，下方的系数图与上一个组件相同。滑块最左端是 λ = 0，即上一个组件里 7 阶的那条曲线；向右每 8 格，λ 增大 10 倍。</div></div>
<script>(function(){{
""" + PLOT + """
const D=__D__;const svg=document.querySelector('#w-i5 svg'),sl=document.getElementById('i5l');
function draw(){const k=+sl.value;base(svg,D);curve(svg,D,D.curves[k]);cars(svg,D);const q=D.rows[k];bars(document.getElementById('i5c'),q.coefs);
 document.getElementById('i5lv').textContent=k===0?'0':(q.lam<100?String(Number(q.lam.toPrecision(2))):Math.round(q.lam));
 document.getElementById('i5n').textContent=q.norm>=1000?q.norm.toExponential(1):q.norm.toFixed(2);
 document.getElementById('i5a').textContent=fmt(q.train);document.getElementById('i5b').textContent=fmt(q.test);}
sl.addEventListener('input',draw);draw();})();</script>""".replace("{{", "{").replace("}}", "}"), d5)
print("widgets ok")
