import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractions import Fraction as Fr
import random
from lifscan.exact import *
random.seed(0)
bad=0
for trial in range(3000):
    T=random.randint(1,40)
    beta=Fr(random.randint(1,10),10)
    theta=Fr(1)
    u_r=Fr(random.choice([0,0,0,-3,2,7]),10)
    I=[Fr(random.randint(-15,15),10) for _ in range(T)]
    u0=Fr(random.randint(-30,12),10)
    us,ss=serial_hard(I,beta,theta,u_r,u0)
    st={}
    u2,s2=tree_scan_states(I,beta,theta,u_r,u0,stats=st)
    M=segment_map(I,beta,theta,u_r,merge=False)
    A,B,x,c=closed_form_segment(I,beta,theta,u_r)
    ok = us==u2 and ss==s2 and M(u0)==us[-1] and (M.A,M.B,M.x,M.c)==(A,B,x,c) and M.pieces<=T+1
    # PWA generic agreement
    P=pwa_segment([pwa_step_hard(i,beta,theta,u_r) for i in I])
    ok = ok and P(u0)==us[-1]
    for uu in [Fr(random.randint(-50,20),10) for _ in range(5)]+x:
        ok = ok and M(uu)==serial_hard(I,beta,theta,u_r,uu)[0][-1]==P(uu)
    if not ok: bad+=1; print("BAD",T,beta,u_r,I,u0); break
print("bad",bad)
