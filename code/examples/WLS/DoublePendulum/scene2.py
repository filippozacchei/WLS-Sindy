from manim import *
import numpy as np

# ---------- DYNAMICS ----------
def f(y, g=9.81, L=1.0, m=1.0, c1=0.07, c2=0.07):
    th1, th2, w1, w2 = y
    s12 = np.sin(th1 - th2)
    c12 = np.cos(th1 - th2)
    denom = 1 + s12**2
    dth1 = w1
    dth2 = w2
    num1 = (-g*(2*np.sin(th1)+np.sin(th1-2*th2)) - 2*s12*(w2**2+w1**2*c12))
    dw1 = num1/(2*L*denom) - (c1/(m*L**2))*w1
    num2 = (2*s12*(2*w1**2+(g/L)*np.cos(th1)+w2**2*c12) - 2*(g/L)*np.sin(th2)*denom)
    dw2 = num2/(2*denom) - (c2/(m*L**2))*w2
    return np.array([dth1,dth2,dw1,dw2])

def rk4_step(y,h):
    k1=f(y); k2=f(y+0.5*h*k1); k3=f(y+0.5*h*k2); k4=f(y+h*k3)
    return y+(h/6)*(k1+2*k2+2*k3+k4)

def simulate(y0,T=12,dt=1/240):
    n=int(T/dt); Y=np.zeros((n,4)); Y[0]=y0
    for i in range(1,n): Y[i]=rk4_step(Y[i-1],dt)
    return np.arange(n)*dt,Y

def tip_positions(th1,th2,L=1.0,origin=np.array([0.,0.])):
    x1=origin[0]+L*np.sin(th1); y1=origin[1]-L*np.cos(th1)
    x2=x1+L*np.sin(th2); y2=y1-L*np.cos(th2)
    return np.array([x1,y1]),np.array([x2,y2])

# ---------- SCENE ----------
class DoublePendulumManyLF(Scene):
    def construct(self):
        rng=np.random.default_rng(0)
        y0=np.array([1.2,-0.9,0.0,0.8]); dt_truth=1/240; T_anim=12
        t,Y=simulate(y0,T=T_anim,dt=dt_truth); th1,th2=Y[:,0],Y[:,1]

        # Sensors
        dt_HF,dt_LF=1/120,1/30; sig_HF,sig_LF=0.003,0.06
        def sample(data,dt_sample,sigma):
            step=int(np.round(dt_sample/(t[1]-t[0])))
            idx=np.arange(0,len(t),step)
            return t[idx], data[idx]+rng.normal(scale=sigma,size=len(idx))
        tsH,th1H=sample(th1,dt_HF,sig_HF); tsH,th2H=sample(th2,dt_HF,sig_HF)

        # 10 low-fidelity runs
        nLF=10
        LF_data=[]
        for i in range(nLF):
            tsL,th1L=sample(th1,dt_LF,sig_LF)
            tsL,th2L=sample(th2,dt_LF,sig_LF)
            LF_data.append((tsL,th1L,th2L))

        # Layout
        left=Rectangle(width=6.5,height=6).set_opacity(0).to_edge(LEFT,buff=0.2)
        right=Rectangle(width=5.0,height=6).set_opacity(0).to_edge(RIGHT,buff=0.2)
        self.add(left,right)
        origin=left.get_center()+np.array([0,1.5,0]); L=1.5

        # 10 LF dots per joint
        lf_dots1=[Dot(radius=0.06,color=RED,fill_opacity=0.3) for _ in range(nLF)]
        lf_dots2=[Dot(radius=0.06,color=RED,fill_opacity=0.3) for _ in range(nLF)]
        for d in lf_dots1+lf_dots2: self.add(d)
        
        # Pendulum
        rod1=Line(color=WHITE,stroke_width=3)
        rod2=Line(color=WHITE,stroke_width=3)
        mass1=Dot(radius=0.08,color=WHITE)
        mass2=Dot(radius=0.09,color=WHITE)
        hf1=Dot(radius=0.06,color=GREEN)
        hf2=Dot(radius=0.06,color=GREEN)
        self.add(rod1,rod2,mass1,mass2,hf1,hf2)

        # Labels
        lbl_hf=VGroup(Dot(radius=0.06,color=GREEN),Text("HF camera",font_size=26)).arrange(RIGHT)
        lbl_lf=VGroup(Dot(radius=0.06,color=RED),Text("LF accelerometer",font_size=26)).arrange(RIGHT)
        VGroup(lbl_hf,lbl_lf).arrange(DOWN,buff=0.2).to_corner(DOWN+LEFT).shift(UP*0.2)
        self.add(lbl_hf,lbl_lf)

        # Axes
        ax=Axes(
            x_range=[0,4,1],y_range=[-np.pi,np.pi,np.pi/2],
            x_length=4.6,y_length=4.0,
            axis_config={"include_tip":False,"stroke_width":2},
        ).move_to(right.get_center()+np.array([0,-0.3,0]))
        self.add(ax)
        title=Text("Joint Angles (HF vs LF ensemble)",font_size=26).next_to(ax,UP,buff=0.2)
        self.add(title)

        # 10 transparent LF lines
        g1_L=[VMobject(color=RED,stroke_width=2,stroke_opacity=0.3) for _ in range(nLF)]
        g2_L=[VMobject(color=RED,stroke_width=2,stroke_opacity=0.3) for _ in range(nLF)]
        for g in g1_L+g2_L: g.set_points_as_corners([ax.c2p(0,0)]); self.add(g)
        
        # HF lines
        g1_H=VMobject(color=GREEN,stroke_width=3)
        g2_H=VMobject(color=GREEN_E,stroke_width=3)
        for g in (g1_H,g2_H): g.set_points_as_corners([ax.c2p(0,0)])
        self.add(g1_H,g2_H)

        # Helper
        def nearest(ts,tau): i=np.searchsorted(ts,tau)-1; return np.clip(i,0,len(ts)-1)
        window=4; fps=60; nF=int(T_anim*fps); dtF=1/fps
        frameT=np.linspace(0,T_anim,nF)
        idx_all=np.minimum((frameT/dt_truth).astype(int),len(t)-1)

        # Animate
        for tau in frameT:
            k_idx=np.searchsorted(frameT,tau)
            th1_k,th2_k=th1[idx_all[k_idx]],th2[idx_all[k_idx]]
            p1,p2=tip_positions(th1_k,th2_k,L=L,origin=origin[:2])
            rod1.put_start_and_end_on(origin,[*p1,0])
            rod2.put_start_and_end_on([*p1,0],[*p2,0])
            mass1.move_to([*p1,0]); mass2.move_to([*p2,0])

            # HF
            iH=nearest(tsH,tau)
            p1H,p2H=tip_positions(th1H[iH],th2H[iH],L=L,origin=origin[:2])
            hf1.move_to([*p1H,0]); hf2.move_to([*p2H,0])

            # Each LF
            for j,(tsL,th1L,th2L) in enumerate(LF_data):
                iL=nearest(tsL,tau)
                p1L,p2L=tip_positions(th1L[iL],th2L[iL],L=L,origin=origin[:2])
                lf_dots1[j].move_to([*p1L,0])
                lf_dots2[j].move_to([*p2L,0])

            # ---- Scrolling window ----
            x_min=max(0.0,tau-window)
            maskH=(tsH<=tau)&(tsH>=x_min)
            h1_pts=[ax.c2p(tsH[j]-x_min,th1H[j]) for j in np.where(maskH)[0]]
            h2_pts=[ax.c2p(tsH[j]-x_min,th2H[j]) for j in np.where(maskH)[0]]
            if len(h1_pts)>1:
                g1_H.set_points_as_corners(h1_pts)
                g2_H.set_points_as_corners(h2_pts)

            for j,(tsL,th1L,th2L) in enumerate(LF_data):
                maskL=(tsL<=tau)&(tsL>=x_min)
                l1_pts=[ax.c2p(tsL[k]-x_min,th1L[k]) for k in np.where(maskL)[0]]
                l2_pts=[ax.c2p(tsL[k]-x_min,th2L[k]) for k in np.where(maskL)[0]]
                if len(l1_pts)>1:
                    g1_L[j].set_points_as_corners(l1_pts)
                    g2_L[j].set_points_as_corners(l2_pts)

            self.wait(1/30)
