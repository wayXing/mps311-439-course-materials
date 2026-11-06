"""Lesson 6 Note and Slides: widget I5, the Titanic passengers on age x fare; drag the depth and watch the tree
cut the plane into smaller boxes while training accuracy rises and cross-validated accuracy falls.
Run (from this folder): python build_widget_i5.py  -> I5.html here (Chinese) and ../note_en/I5.html (English)
Data as build_numbers_v4.py: seaborn-data titanic.csv, rows without age dropped, stratified 3:1 split, random_state=0;
only two features (age, fare) so the boxes can be drawn; KFold(5, shuffle, 0)."""
import json, os, re, subprocess, tempfile, shutil
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
B, G, R = "#0072B2", "#00795A", "#C84B00"
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
ship = pd.read_csv(URL)[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].dropna()
X = ship[["age", "fare"]]; y = ship["survived"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
kf = KFold(5, shuffle=True, random_state=0)
AGE, FARE = (0.0, 81.0), (0.0, 520.0)
def boxes(t):
    T = t.tree_; out = []
    def go(i, a0, a1, f0, f1):
        if T.children_left[i] < 0:
            out.append([round(a0, 2), round(a1, 2), round(f0, 2), round(f1, 2), int(np.argmax(T.value[i][0]))]); return
        c = T.threshold[i]
        if T.feature[i] == 0: go(T.children_left[i], a0, min(a1, c), f0, f1); go(T.children_right[i], max(a0, c), a1, f0, f1)
        else: go(T.children_left[i], a0, a1, f0, min(f1, c)); go(T.children_right[i], a0, a1, max(f0, c), f1)
    go(0, AGE[0], AGE[1], FARE[0], FARE[1]); return out
depths = {}
for k, d in enumerate(list(range(1, 11)) + [None], start=1):
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    depths[str(k)] = dict(label=str(d) if d else "∞", train=round(t.score(Xtr, ytr), 3),
                          cv=round(cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), Xtr, ytr, cv=kf).mean(), 3),
                          leaves=int(t.get_n_leaves()), boxes=boxes(t))
data = dict(age=[round(v, 1) for v in Xtr.age], fare=[round(v, 2) for v in Xtr.fare], s=[int(v) for v in ytr], depths=depths)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "boxes"} for k, v in depths.items()}, open(os.path.join(HERE, "i5_numbers.json"), "w"), indent=1)
CSS = """<style>
.wbox{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:1.2em 0;background:#fafbfc;font-size:.95em}
.wbox svg{display:block;max-width:100%;height:auto}
.wbox .wrow{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin-top:8px}
.wbox .wn{font-variant-numeric:tabular-nums;font-weight:700}
.wbox .cap{color:#555;font-size:.9em;margin-top:6px}
.wbox input[type=range]{width:180px}
</style>"""
TXT = {"zh": dict(depth="最大深度", leaves="叶子数", train="训练准确率", cv="5 折交叉验证", surv="生还", died="遇难", green="浅绿底色：这一块判为生还",
                  xl="年龄", yl="票价（对数刻度）",
                  cap="535 名训练乘客，只用年龄和票价两个特征，方便把树切出的矩形画出来。拖动最大深度：矩形越切越碎，训练准确率一路上升，交叉验证先升后降。最右一格是不限深度。"),
       "en": dict(depth="max depth", leaves="leaves", train="training accuracy", cv="5-fold CV", surv="survived", died="died", green="light green: predicted survived",
                  xl="age", yl="fare (log scale)",
                  cap="535 training passengers, two features only (age and fare) so the boxes can be drawn. Drag the depth: the boxes get smaller, training accuracy keeps rising, cross-validation rises and then falls. The last step is no depth limit.")}
for lang, out in [("zh", HERE), ("en", os.path.join(HERE, "..", "note_en"))]:
    t = TXT[lang]
    body = f"""<div class="wbox" id="w-i5">
<svg viewBox="0 0 520 400" style="max-width:520px"></svg>
<div class="wrow"><label>{t['depth']} <input type="range" id="i5d" min="1" max="11" step="1" value="1"> <span class="wn" id="i5dv"></span></label></div>
<div class="wrow"><span>{t['leaves']} <span class="wn" id="i5n"></span></span><span style="color:{B}">{t['train']} <span class="wn" id="i5t"></span></span><span style="color:{G}">{t['cv']} <span class="wn" id="i5c"></span></span></div>
<div class="wrow" style="font-size:.9em;color:#444"><span><span style="color:{B}">●</span> {t['surv']}</span><span><span style="color:#777">○</span> {t['died']}</span><span>{t['green']}</span></div>
<div class="cap">{t['cap']}</div></div>
<script>(function(){{
const NS='http://www.w3.org/2000/svg';function el(p,tg,a,txt){{const e=document.createElementNS(NS,tg);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;p.appendChild(e);return e;}}
const D=__D__;const svg=document.querySelector('#w-i5 svg'),sl=document.getElementById('i5d');
const L=50,Rr=510,T=8,Bt=350,A0=0,A1=81,ly=f=>Math.log10(f+1),F0=ly(0),F1=ly(520);
const sx=a=>L+(a-A0)/(A1-A0)*(Rr-L),sy=f=>Bt-(ly(f)-F0)/(F1-F0)*(Bt-T);
function draw(){{const q=D.depths[sl.value];svg.innerHTML='';
 q.boxes.forEach(b=>{{if(b[4]===1)el(svg,'rect',{{x:sx(b[0]),y:sy(b[3]),width:sx(b[1])-sx(b[0]),height:sy(b[2])-sy(b[3]),fill:'{G}','fill-opacity':0.16}});
  el(svg,'rect',{{x:sx(b[0]),y:sy(b[3]),width:sx(b[1])-sx(b[0]),height:sy(b[2])-sy(b[3]),fill:'none',stroke:'#777','stroke-width':0.6}});}});
 D.age.forEach((a,k)=>{{const x=sx(a),yv=sy(D.fare[k]);if(D.s[k])el(svg,'circle',{{cx:x,cy:yv,r:3.2,fill:'{B}','fill-opacity':0.85}});else el(svg,'circle',{{cx:x,cy:yv,r:2.8,fill:'none',stroke:'#666','stroke-width':1}});}});
 el(svg,'line',{{x1:L,x2:Rr,y1:Bt,y2:Bt,stroke:'#888'}});el(svg,'line',{{x1:L,x2:L,y1:T,y2:Bt,stroke:'#888'}});
 [0,20,40,60,80].forEach(v=>el(svg,'text',{{x:sx(v),y:Bt+15,'text-anchor':'middle','font-size':12,fill:'#444'}},v));
 [0,5,10,20,50,100,300].forEach(v=>el(svg,'text',{{x:L-6,y:sy(v)+4,'text-anchor':'end','font-size':12,fill:'#444'}},v));
 el(svg,'text',{{x:(L+Rr)/2,y:Bt+34,'text-anchor':'middle','font-size':13,fill:'#333'}},'{t['xl']}');
 el(svg,'text',{{x:12,y:(T+Bt)/2,'font-size':13,fill:'#333',transform:'rotate(-90 12 '+(T+Bt)/2+')','text-anchor':'middle'}},'{t['yl']}');
 document.getElementById('i5dv').textContent=q.label;document.getElementById('i5n').textContent=q.leaves;
 document.getElementById('i5t').textContent=q.train.toFixed(3);document.getElementById('i5c').textContent=q.cv.toFixed(3);}}
sl.addEventListener('input',draw);draw();}})();</script>"""
    html = "```{=html}\n" + CSS + "\n" + body.replace("__D__", json.dumps(data, ensure_ascii=False)) + "\n```\n"
    if lang == "en" and re.findall(r"[一-鿿]", html):
        raise SystemExit("Chinese left in the English I5")
    open(os.path.join(out, "I5.html"), "w", encoding="utf8").write(html)
    if shutil.which("node"):
        js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh: fh.write(js)
        res = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True); os.remove(fh.name)
        if res.returncode: raise SystemExit(res.stderr[:300])
print({k: (v["label"], v["train"], v["cv"], v["leaves"]) for k, v in depths.items()})
