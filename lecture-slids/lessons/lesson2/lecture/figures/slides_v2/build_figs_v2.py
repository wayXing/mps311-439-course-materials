import re
"""Lesson 2 deck v2 figures. Rules fixed before looking at results:
 small set = first 10 rows of the training split (random_state=42);
 hook car = test car whose |error| under the 313-car hp line is closest to the test RMSE.
 Spaghetti and scatter use 20 / 200 random 10-car sets from the 313 training cars (seed 1), no selection."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.patches import Rectangle
B,G,R="#0072B2","#00795A","#C84B00"; GREY="#6b6b6b"
plt.rcParams.update({"font.family":["DejaVu Sans"],"svg.fonttype":"path","axes.spines.top":False,"axes.spines.right":False,"font.size":13,"axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13})
df=pd.read_csv("data/auto_mpg.csv");F6=["cylinders","displacement","horsepower","weight","acceleration","model_year"]
Xtr,Xte,ytr,yte=train_test_split(df[F6],df.mpg,test_size=0.2,random_state=42)
hp,y=Xtr.horsepower.values,ytr.values; ht,yt=Xte.horsepower.values,yte.values
rm=lambda a,b:float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
x10,y10=hp[:10],y[:10]
w10,b10=np.polyfit(x10,y10,1); w313,b313=np.polyfit(hp,y,1)
tr10,te10=rm(y10,b10+w10*x10),rm(yt,b10+w10*ht)
rmse313=rm(yt,b313+w313*ht)
out={"x10":x10.tolist(),"y10":y10.tolist(),"w10":w10,"b10":b10,"w313":w313,"b313":b313,"tr10":tr10,"te10":te10,"rmse313_test":rmse313,"rmse313_train":rm(y,b313+w313*hp)}
def save(fig,n): fig.savefig(f"figs/{n}.svg",bbox_inches="tight"); plt.close(fig)
# hook car
pred=b313+w313*ht; err=np.abs(yt-pred); hi=int(np.argmin(np.abs(err-rmse313)))
hook=dict(hp=float(ht[hi]),actual=float(yt[hi]),pred=float(pred[hi]),err=float(err[hi]),idx=int(Xte.index[hi]),row=Xte.iloc[hi].to_dict()); out["hook"]=hook
# S2 opening scatter
fig,ax=plt.subplots(figsize=(9,4.3))
ax.scatter(x10,y10,color=B,s=110,zorder=3)
ax.axvline(hook["hp"],color=GREY,ls="--",lw=2)
ax.text(hook["hp"]+3,41.5,"New car: 140 hp\nmpg = ?",color=B,weight="bold",fontsize=15,va="top")
ax.set_xlim(50,200);ax.set_ylim(10,45);ax.set_xlabel("Horsepower (hp)");ax.set_ylabel("mpg")
save(fig,"S2")
# A/B lines
LA,LB=(39,-0.15),(33,-0.09)
def lines(name,resid):
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),sharey=True); xl=np.array([55,195])
    for ax,(nm,(b,w)) in zip(axs,[("Line A",LA),("Line B",LB)]):
        ax.plot(xl,b+w*xl,color=G,lw=3)
        if resid:
            for xi,yi in zip(x10,y10): ax.plot([xi,xi],[yi,b+w*xi],color=R,lw=2.6)
        ax.scatter(x10,y10,color=B,s=90,zorder=3)
        m=np.mean((y10-(b+w*x10))**2)
        ax.set_title(f"{nm}: RMSE {np.sqrt(m):.2f} mpg" if resid else nm,fontsize=15,weight="bold")
        ax.set_xlabel("Horsepower (hp)");ax.set_xlim(50,200);ax.set_ylim(10,45)
        out[nm+"_mse"]=float(m)
    axs[0].set_ylabel("mpg"); fig.tight_layout(); save(fig,name)
lines("F4a",False); lines("F4b",True)
# F5 bowl for the 10 cars
from mpl_toolkits.mplot3d import Axes3D  # noqa
bb=np.linspace(20,58,200);ww=np.linspace(-0.40,0.06,200);BB,WW=np.meshgrid(bb,ww)
J=np.mean((y10[None,None,:]-(BB[...,None]+WW[...,None]*x10[None,None,:]))**2,axis=-1)
LJ=np.log10(J);lev=np.linspace(LJ.min(),LJ.max(),15)
fig=plt.figure(figsize=(11,4.4));ax=fig.add_subplot(1,2,1)
ax.contourf(BB,WW,LJ,levels=lev,cmap="Oranges",alpha=0.85,rasterized=True);ax.contour(BB,WW,LJ,levels=lev,colors="white",linewidths=0.4,rasterized=True)
ax.scatter([b10],[w10],color=R,s=130,zorder=4,marker="*")
for lab,(bq,wq) in {"A":LA,"B":LB}.items():
    ax.scatter([bq],[wq],color=G,s=60,zorder=4);ax.text(bq+0.8,wq+0.008,lab,color=G,weight="bold",fontsize=14)
ax.set_xlabel("Intercept b");ax.set_ylabel("Slope w");ax.set_title("From above: contour lines",fontsize=14)
a3=fig.add_subplot(1,2,2,projection="3d");band=np.floor((LJ-LJ.min())/(LJ.max()-LJ.min())*14.999)/14
a3.plot_surface(BB,WW,J,facecolors=plt.cm.Oranges(0.05+0.85*band),rstride=2,cstride=2,linewidth=0,antialiased=False,shade=False,rasterized=True)
a3.scatter([b10],[w10],[np.mean((y10-(b10+w10*x10))**2)],color=R,s=120,marker="*",zorder=10);a3.view_init(elev=32,azim=-58)
a3.set_xlabel("b");a3.set_ylabel("w");a3.set_zlabel("MSE",labelpad=12);a3.set_title("From the side: the loss is a bowl",fontsize=14)
fig.tight_layout();save(fig,"F5")
# F13 spaghetti
rng=np.random.default_rng(1)
fig,ax=plt.subplots(figsize=(9,4.6));xl=np.array([45,230]);ax.scatter(hp,y,color="#c9ced6",s=14,lw=0,zorder=1)
k=0;slopes=[]
while k<20:
    i=rng.choice(313,10,replace=False)
    if np.ptp(hp[i])<30: continue
    w,b=np.polyfit(hp[i],y[i],1);ax.plot(xl,b+w*xl,color=G,lw=1.6,alpha=0.35,zorder=2);k+=1;slopes.append(w)
ax.plot(xl,b10+w10*xl,color=G,lw=3.6,zorder=4,label="Our 10 cars");ax.scatter(x10,y10,color=B,s=60,zorder=5)
ax.plot(xl,b313+w313*xl,color="#172033",lw=2.6,ls="--",zorder=4,label="All 313 training cars")
ax.set_xlim(45,230);ax.set_ylim(5,47);ax.set_xlabel("Horsepower (hp)");ax.set_ylabel("mpg");ax.legend(frameon=False,fontsize=13,loc="upper right")
out["spaghetti_slopes_range"]=[float(min(slopes)),float(max(slopes))];save(fig,"F13")
# F14 train vs test over 200 sets
rng=np.random.default_rng(1);tr=[];te=[]
while len(tr)<200:
    i=rng.choice(313,10,replace=False)
    if np.ptp(hp[i])<30: continue
    w,b=np.polyfit(hp[i],y[i],1);tr.append(rm(y[i],b+w*hp[i]));te.append(rm(yt,b+w*ht))
tr,te=np.array(tr),np.array(te);frac=float(np.mean(te>tr));out["f14"]=dict(frac_test_gt_train=frac,med_train=float(np.median(tr)),med_test=float(np.median(te)),n_clipped=int(np.sum(te>9)))
fig,ax=plt.subplots(figsize=(6.2,5.2));ax.plot([2,9],[2,9],color=GREY,lw=1.5,ls="--")
ax.scatter(tr,np.minimum(te,9),color=G,alpha=0.35,s=34,lw=0)
ax.scatter([tr10],[te10],color=R,s=160,zorder=5,marker="*")
ax.annotate("our 10 cars",(tr10,te10),(tr10+0.5,te10-1.3),color=R,fontsize=13,weight="bold",arrowprops=dict(arrowstyle="-",color=R))
ax.set_xlim(2,9);ax.set_ylim(2,9);ax.set_xlabel("RMSE on the 10 cars used to fit (mpg)");ax.set_ylabel("RMSE on the 79 test cars (mpg)")
ax.text(2.25,8.55,f"{frac*100:.0f}% of the dots are above the line",color=GREY,fontsize=13)
ax.text(3.6,2.25,"Dots above 9 are drawn at the top edge",color=GREY,fontsize=11)
save(fig,"F14")
# F17 back to the car
fig,ax=plt.subplots(figsize=(9,2.9));p,a=hook["pred"],hook["actual"]
ax.fill_betweenx([-0.18,0.18],p-rmse313,p+rmse313,color=G,alpha=0.15,lw=0)
ax.plot([p,a],[0,0],color=R,lw=3.5,solid_capstyle="butt");ax.scatter([p],[0],color=G,s=150,zorder=3);ax.scatter([a],[0],color=B,s=150,zorder=3)
ax.text(p+0.3,0.3,f"Model prediction {p:.1f}",ha="left",color=G,weight="bold",fontsize=15);ax.text(a-0.3,0.3,f"Actual {a:.1f}",ha="right",color=B,weight="bold",fontsize=15)
ax.text((p+a)/2,-0.4,f"Error {abs(p-a):.1f}",ha="center",color=R,weight="bold",fontsize=15)
ax.text(p+rmse313,-0.24,f"Typical error {rmse313:.2f}",ha="right",color=G,fontsize=13,va="top")
ax.set_xlim(8,26);ax.set_ylim(-0.85,0.75);ax.set_yticks([]);ax.set_xlabel("mpg");ax.spines["left"].set_visible(False)
save(fig,"F17")
json.dump(out,open("figs/numbers.json","w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))

# F2 for our cars
rows=Xtr.iloc[:4].copy(); rows["mpg"]=ytr.iloc[:4].values
heads=["cyl","disp","hp","wt","acc","year","mpg"]
fig,ax=plt.subplots(figsize=(7.4,2.5)); ax.axis("off"); ax.set_xlim(0,8); ax.set_ylim(0,5.2)
ax.add_patch(Rectangle((0.9,0.35),5.7,3.85,color=B,alpha=0.08,lw=0)); ax.add_patch(Rectangle((6.65,0.35),1.2,3.85,color=B,alpha=0.20,lw=0))
ax.text(3.75,4.9,"Feature x (used to predict)",ha="center",color=B,weight="bold"); ax.text(7.25,4.9,"Target y",ha="center",color=B,weight="bold")
ax.plot([0.95,6.55],[4.62,4.62],color=B,lw=1.5); ax.plot([6.7,7.8],[4.62,4.62],color=B,lw=1.5)
xpos=[1.3,2.35,3.4,4.45,5.5,6.2,7.25]; ax.text(0.45,3.85,"Car",ha="center",color=GREY,weight="bold")
for xp,h in zip(xpos,heads): ax.text(xp,3.85,h,ha="center",weight="bold")
for i,(_,r) in enumerate(rows.iterrows()):
    yy=3.05-i*0.75; ax.text(0.45,yy,str(i+1),ha="center",color=GREY)
    vals=[f"{int(r.cylinders)}",f"{r.displacement:.0f}",f"{r.horsepower:.0f}",f"{int(r.weight)}",f"{r.acceleration:.1f}",f"{int(r.model_year)}",f"{r.mpg:.1f}"]
    for xp,v in zip(xpos,vals): ax.text(xp,yy,v,ha="center")
ax.text(4.0,0.0,"Each row is one car",ha="center",color=GREY,fontsize=10)
save(fig,"F2"); print("car2 weight",rows.iloc[1].weight, rows.iloc[1].to_dict())
# I3 widget for the ten cars
src=open("figures/I3.html").read()
D=json.dumps({"hp":x10.tolist(),"mpg":y10.tolist()})
src=re.sub(r'const D=\{.*?\};',"const D="+D+";",src,count=1)
src=src.replace("const X0=120,X1=175,Y0=10,Y1=24","const X0=50,X1=200,Y0=10,Y1=45").replace("const HX=[122,173],init=[21,13.5]","const HX=[52,198],init=[33,22]")
src=src.replace("for(let x=120;x<=170;x+=10)","for(let x=50;x<=200;x+=25)").replace("for(let y=10;y<=24;y+=2)","for(let y=10;y<=45;y+=5)")
src=src.replace("the three cars best","the ten cars best")
open("figs/I3.html","w").write(src)
