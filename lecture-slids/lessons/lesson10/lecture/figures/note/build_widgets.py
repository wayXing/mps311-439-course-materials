"""Lesson 10 Note: self-contained HTML widgets (no CDN), data from numbers.json.
Run (from this folder):  python build_widgets.py            -> Chinese widgets here
                         python build_widgets.py --lang en  -> English widgets in ../note_en/
I1  a 3x3 filter slides over a digit; the feature map (after ReLU) shows where it lights up
I2d shift a digit 0 to 4 pixels: validation accuracy of the fully connected network (Section 1)
I2  the same with the CNN as well (Section 5)
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
.wbox input[type=range]{width:220px}
.wbox button{font:inherit;padding:3px 10px;border:2px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
.wbox button.on{border-color:#00795A;color:#00795A;font-weight:700}
</style>"""
EL = """const NS='http://www.w3.org/2000/svg';
function el(p,t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}
function grey(v){const c=Math.round(255*(1-Math.max(0,Math.min(1,v))));return 'rgb('+c+','+c+','+c+')';}"""


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


wrap("I1", """<div class="wbox" id="w-i1">
<div class="wrow"><span>数字：</span><button data-d="0">1</button><button data-d="1">4</button><button data-d="2">7</button>
<span style="margin-left:14px">滤波器：</span><button data-f="v">竖边</button><button data-f="h">横边</button><button data-f="d">斜边</button><button data-f="b">全是 1</button></div>
<div class="wrow" style="align-items:flex-start;gap:16px">
<div><div>图（点一下选位置）</div><svg id="i1img" viewBox="0 0 280 280" style="width:280px"></svg></div>
<div><div>3×3 滤波器（点格子改：1 → 0 → −1）</div><svg id="i1f" viewBox="0 0 150 150" style="width:150px"></svg>
<div style="margin-top:10px">这个位置的值：<span class="wn" id="i1v"></span></div><div class="wrow"><button id="i1p">滑过去</button></div></div>
<div><div>特征图（26×26，负数记 0）</div><svg id="i1m" viewBox="0 0 260 260" style="width:260px"></svg></div></div>
<div class="cap">滤波器的 9 个权重乘上图里 3×3 一小块的 9 个像素，再加起来，得到特征图上的一个数。同一个滤波器从左上角滑到右下角，就得到整张特征图：哪里黑，就是这个滤波器在那里“亮”。</div></div>
<script>(function(){
__EL__
const D=__D__;const P={v:[[1,0,-1],[1,0,-1],[1,0,-1]],h:[[1,1,1],[0,0,0],[-1,-1,-1]],d:[[0,1,1],[-1,0,1],[-1,-1,0]],b:[[1,1,1],[1,1,1],[1,1,1]]};
let img=D[2].img,w=P.h.map(r=>r.slice()),pos=[10,10],dig=2,fil='h',timer=null;
const si=document.getElementById('i1img'),sf=document.getElementById('i1f'),sm=document.getElementById('i1m');
function fmap(){const M=[];for(let i=0;i<26;i++){M.push([]);for(let j=0;j<26;j++){let s=0;for(let a=0;a<3;a++)for(let b=0;b<3;b++)s+=w[a][b]*img[i+a][j+b];M[i].push(Math.max(0,s));}}return M;}
function draw(){si.innerHTML='';sf.innerHTML='';sm.innerHTML='';
 for(let i=0;i<28;i++)for(let j=0;j<28;j++)el(si,'rect',{x:j*10,y:i*10,width:10.5,height:10.5,fill:grey(img[i][j])});
 el(si,'rect',{x:pos[1]*10,y:pos[0]*10,width:30,height:30,fill:'none',stroke:'#00795A','stroke-width':3});
 for(let a=0;a<3;a++)for(let b=0;b<3;b++){const v=w[a][b];const g=el(sf,'g',{style:'cursor:pointer'});
  el(g,'rect',{x:b*50,y:a*50,width:48,height:48,fill:v>0?'#f4c7b8':v<0?'#bcd7ea':'#fff',stroke:'#888'});
  el(g,'text',{x:b*50+24,y:a*50+31,'text-anchor':'middle','font-size':20},v);
  g.addEventListener('click',()=>{w[a][b]=v===1?0:v===0?-1:1;fil='';draw();});}
 const M=fmap();let mx=0;M.forEach(r=>r.forEach(v=>{if(v>mx)mx=v;}));
 for(let i=0;i<26;i++)for(let j=0;j<26;j++)el(sm,'rect',{x:j*10,y:i*10,width:10.5,height:10.5,fill:grey(M[i][j]/3)});
 el(sm,'rect',{x:pos[1]*10,y:pos[0]*10,width:10,height:10,fill:'none',stroke:'#00795A','stroke-width':2.5});
 document.getElementById('i1v').textContent=M[pos[0]][pos[1]].toFixed(2);
 document.querySelectorAll('#w-i1 button[data-d]').forEach(b=>b.classList.toggle('on',+b.dataset.d===dig));
 document.querySelectorAll('#w-i1 button[data-f]').forEach(b=>b.classList.toggle('on',b.dataset.f===fil));}
si.addEventListener('click',e=>{const r=si.getBoundingClientRect();const j=Math.floor((e.clientX-r.left)/r.width*28),i=Math.floor((e.clientY-r.top)/r.height*28);
 pos=[Math.max(0,Math.min(25,i-1)),Math.max(0,Math.min(25,j-1))];draw();});
document.querySelectorAll('#w-i1 button[data-d]').forEach(b=>b.addEventListener('click',()=>{dig=+b.dataset.d;img=D[dig].img;draw();}));
document.querySelectorAll('#w-i1 button[data-f]').forEach(b=>b.addEventListener('click',()=>{fil=b.dataset.f;w=P[fil].map(r=>r.slice());draw();}));
document.getElementById('i1p').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;return;}pos=[0,0];
 timer=setInterval(()=>{pos=[pos[0],pos[1]+1];if(pos[1]>25){pos=[pos[0]+1,0];}if(pos[0]>25){clearInterval(timer);timer=null;pos=[25,25];}draw();},15);});
draw();})();</script>""", N["widget_images"])

seven = [w for w in N["widget_images"] if w["digit"] == 7][0]["img"]
I2 = """<div class="wbox" id="w-__ID__">
<div class="wrow"><span>整张图右移：</span><input type="range" id="__ID__k" min="0" max="4" step="1" value="0"><span class="wn" id="__ID__kv"></span><span>个像素</span></div>
<div class="wrow" style="align-items:center;gap:20px"><svg id="__ID__img" viewBox="0 0 168 168" style="width:168px"></svg>
<svg id="__ID__b" viewBox="0 0 420 __H__" style="width:420px"></svg></div>
<div class="cap">验证集的 10,000 张图全部右移，左边空出的列补成白色。网络没有重新训练，只是重新测一次准确率。</div></div>
<script>(function(){
__EL__
const D=__D__;const s=document.getElementById('__ID__k'),si=document.getElementById('__ID__img'),sb=document.getElementById('__ID__b');
function draw(){const k=+s.value;document.getElementById('__ID__kv').textContent=k;si.innerHTML='';sb.innerHTML='';
 for(let i=0;i<28;i++)for(let j=0;j<28;j++){const v=j-k>=0?D.img[i][j-k]:0;el(si,'rect',{x:j*6,y:i*6,width:6.4,height:6.4,fill:grey(v)});}
 el(si,'rect',{x:0,y:0,width:168,height:168,fill:'none',stroke:'#888'});
 D.rows.forEach((r,n)=>{const y=10+n*46,v=r.acc[k],x0=130,W=225;el(sb,'text',{x:x0-8,y:y+21,'text-anchor':'end','font-size':14},r.name);
  el(sb,'rect',{x:x0,y:y,width:W*v,height:30,fill:r.col});el(sb,'text',{x:x0+W*v+6,y:y+21,'font-size':14,'font-weight':700},v.toFixed(3));});}
s.addEventListener('input',draw);draw();})();</script>"""
for name, rows in [("I2d", [dict(name="全连接网络", acc=N["dense"]["shift"], col="#00795A")]),
                   ("I2", [dict(name="全连接网络", acc=N["dense"]["shift"], col="#6b6b6b"), dict(name="CNN", acc=N["cnn"]["shift"], col="#00795A")])]:
    body = I2.replace("__ID__", name.lower()).replace("__H__", str(20 + 46 * len(rows)))
    wrap(name, body, dict(img=seven, rows=rows))
print("widgets ok")
