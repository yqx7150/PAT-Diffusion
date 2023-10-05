
import numpy as np
import torch
# import scipy
from scipy import linalg
import scipy.io as scio
import math
##

def rank_estimator_adaptive(k,iter,est_rank,reschg,tol,rk,rank_max,itr_rank,R,rk_jump, minitr_reduce_rank, maxitr_reduce_rank,rank_min,Zfull,iprint,alf,rk_inc,save_res,X,Y,m,n):
    est_rank=torch.tensor(est_rank).cuda()
    if est_rank == 1:
        ###no###
        
        dR = abs(np.diag(R))
        drops = dR[0:-2]/dR[1:-1]
        dmx,imx = max(drops)
        rel_drp = (k-1)*dmx/(sum(drops)-dmx)


        #???
        if  ((rel_drp > rk_jump) & (itr_rank > minitr_reduce_rank)) | (itr_rank > maxitr_reduce_rank):
            
            rk = max([imx, math.floor(0.1*k), rank_min])
            X = X[:,1:rk]; Y = Y[1:rk,:]
            if Zfull:
               

                Z =np.dot(X,Y)
                xx=Z.shape[0]
                yy=Z.shape[1]
                Z=np.reshape(Z.T,[Z.shape[0]*Z.shape[1],1])
                Res=data-Z[Known]
                Z[Known]=data+alf*Res
                Z=np.reshape(Z,[yy,xx])
                Z=Z.T
            
            else:#不执行未修改
                
                Res = data - partXY(X.T,Y,Ik,Jk,L)
                updateSval(S, (alf+1)*Res, L)
            res = np.linalg.norm(Res)
            est_rank = 0
            itr_rank = 0
            if iprint >= 2:
                ##format  未完全设置格式精度
                print('it: {5i}, rel_drp: {3.2e}, est_rank: {d},  Rank estimate changed from {i} to {i}\n',format(iter, rel_drp, est_rank, k,rk))
                
    elif (est_rank == 2) &( reschg < 10*tol) & (rk < rank_max )& (itr_rank > 1) :#进
        
        if rk < 50: 
            rinc = rk_inc
        else  : ###yes###
            rinc = 2*rk_inc
        rk = min(rk + rinc, rank_max)
        rkr_id = True
        if rk > k:###yes###
            if save_res == 1:  ###no###
                # save(strcat('LM-Med-r', num2str(rk),'max', num2str(rank_max),  '.mat'), 'X', 'Y')  
                str='LM-Med-r'+str(rk)+'max'+str(rank_max)+ '.mat'
                scio.savemat(str,'X','Y')

            # X = [X, zeros(m, rk-k)]; Y = [Y; zeros(rk-k,n)]; 
            # X=np.concatenate([X, np.zeros((n,rk-k))], axis=0)
            # Y=np.concatenate([Y, np.zeros((rk-k,n))], axis=1)
            
            
            
            X=torch.hstack((X,torch.zeros((m, rk-k),dtype=torch.float64).cuda()))
            Y=torch.vstack((Y,torch.zeros((rk-k,n),dtype=torch.float64).cuda()))
            itr_rank = 0
            if iprint >= 2:  #####no####

                ###format() #未完全设置精度{}中的值
                print('it: {5i}, reschg: {3.2e}, Rank estimate changed from {i} to {i}\n',format(iter,reschg, k,rk)) 
    return k,est_rank,reschg,tol,rk,rank_max,itr_rank,R,rk_jump, minitr_reduce_rank, maxitr_reduce_rank,rank_min,Zfull,iprint,alf,rk_inc,save_res,X,Y,m,n         


def lmafit_mc_adp(m,n,k,Known,data,opts):
    
    # Known=Known-1
    
    
    L = len(data)
    tol = 1.25e-4
    maxit = 50
    iprint = 1
    Zfull = (L/(m*n) > 0.2 ) or k > .02*(min(m,n)) or m*n < 5e5   ##.02????条件：或，最小值min，python中的*对应
    DoQR = True   
    
    

    est_rank = 1
    rank_max =  min(m,n)-1
    rank_min =  1
    rk_inc = 1
    rk_jump = 10
    init = 0
    save_res = 0
    if 'tol' in opts:
        tol =  opts['tol']
    if 'maxit' in opts:
        maxit =  opts['maxit']    
    if 'print' in opts:
        iprint =  opts['print']
    if 'Zfull' in opts:
        Zfull =  opts['Zfull']
    if 'DoQR' in opts:
        DoQR =  opts['DoQR']
    if 'est_rank' in opts:
        est_rank =  opts['est_rank']
    if 'rank_max' in opts:
        rank_max =  opts['rank_max']
    if 'rank_min' in opts:
        rank_min =  opts['rank_min']
    if 'rk_inc' in opts:
        rk_inc =  opts['rk_inc']
    if 'rk_jump' in opts:
        rk_jump =  opts['rk_jump']
    if 'init' in opts:
        init =  opts['init']
    if 'save_res' in opts:
        save_res =  opts['save_res']
    
    
    
    reschg_tol = 0.5*tol
    rk = k
    if est_rank == 1:
        rank_max = min(rank_max, k)

    ##将linsolve中属性sym 和 posdef 设成true 
    # linopts.SYM = True
    # linopts.POSDEF = True
    
    #print(torch.linalg.norm(data),data.shape)
    #print(np.linalg.norm(data.cpu().numpy()))


    datanrm = max(1,torch.linalg.norm(data)) 
    objv = torch.zeros((maxit,1),dtype=torch.float64)   
    RR =torch.ones((maxit,1),dtype=torch.float64) 
    
    
    
   
    if iprint==1:#无
        print('Iteration:     ')
    if iprint==2:#无
        print('\nLMafit_mc: Zfull = %i, DoQR = %i\n',Zfull,DoQR)
    
    
    
    data[data==0] = 2.2204e-16 # data中等于0的值为eps  matlab中的eps,这里先表示数值大小
    data_tran = False
    if Zfull:#yes

        
        # if isstruct(Known), Known = sub2ind([m n], Known.Ik,Known.Jk); ###no###matlab将下标转化索引#  无  未执行,未完全修改
        # Z = np.zeros((m,n))
        Z=torch.zeros((m*n,1),dtype=torch.float64).cuda()#先将(m,n)矩阵展开成一维--赋值--reshape(n,m)--转置
        Z[Known] = data  
        Z=torch.reshape(Z,(n,m))
        

        
        Z=Z.T
        
        
        
        
        
        
       
    else:  ######nononono##### #不执行未修改
        
        
        if isnumeric(Known):   
            [Ik,Jk] = ind2sub([m ,n],Known);##matlab将索引改为下标
        else:
            
            if isstruct(Known)  :
                Ik = Known.Ik
                Jk = Known.Jk
        
        S = sparse(Ik, Jk, data, m, n)
        [Ik, Jk, data] = find(S);  data = data.t####转置


    if m>n:
        tmp = m
        m = n
        n = tmp
        data_tran = True   
        if Zfull:  ##yes
            
            Z=Z.T
            M=Z
            M=torch.reshape(M.T,[m*n,1])
            Known=torch.where(M!=0)[0]
            data=M[Known]
            
           
            
            
        else:####no##### #不执行未完全修改
            
            
            S = S.T   
            Ik, Jk= np.where(S!=0)
            data=S[Ik,Jk]
            
    if init == 0:##进
        X = torch.zeros((m,k),dtype=torch.float64).cuda()
        Y = torch.eye(k,n).to(torch.float64).cuda()
        Res = data
        res = datanrm
        
        
        
        
    else:   ####nonon####          
        
        X = opts.X    #opts中没有xy
        Y = opts.Y
        opts.X = []  #[]list
        opts.Y = []
        if Zfull:#不执行未完全修改
            Z = X*Y
            Res = data - Z(Known)
            Z[Known] = data  ### 简单表示，但错误
        else:#不执行未修改
            Res = data - partXY(X.T,Y,Ik,Jk,L)
            updateSval(S, Res, L)
        res =np.linalg.norm(Res)


    ##parameters for alf
    alf = 0
    increment = 1
    itr_rank = 0
    minitr_reduce_rank = 5
    maxitr_reduce_rank = 50
    
    mm=0
    
    for iter in range(maxit):
        print(iter)
        
        itr_rank = itr_rank + 1
        Xo = X
        Yo = Y
        Res0 = Res
        res0 = res
        alf0x = alf
        if Zfull:
            Zo = Z
            # X = Z*Y.T
            

            
            X=torch.mm(Z,Y.T)
            
            

            
            if est_rank == 1:###no###
                print('est_rank == 1')
                X,R,E = np.linalg.qr(X,mode='economic')
                # Y = X.t*Z
                Y=np.dot(X.T,Z)
            elif DoQR:#进
                #X ,R  = torch.linalg.qr(X,mode='reduced')#这里用的是from scipy import linalg
                X ,R  = torch.qr(X,some=True)
                # print(np.linalg.qr(X,mode='economic'))
                # Y = X.T*Z
                Y = torch.mm(X.T,Z)
                mm=mm+1
                
                
                
                

            else:####no####
                
                Xt = X.T
                
                # Y = linsolve(Xt*X,Xt*Z,linopts)  #导入sympy包  没有修改
                Y=linsolve(np.dot(Xt,X),np.dot(Xt,Z),linopts)
            # Z = X*Y
            Z=torch.mm(X,Y)
            M=Z
            M=torch.reshape(M.T,[M.shape[0]*M.shape[1],1])

            Res = data - M[Known]
            
            
            
            
        else:####NO####
            
            Yt = Y.T
            # X = S*Yt + X*(Y*Yt)
            X=np.dot(S,Yt)+np.dot(X,np.dot(Y,Yt))
            if est_rank == 1:
                X,R,E =linalg.qr(X,mode='economic')
                Xt = X.T
                # Y = Xt*S + (Xt*Xo)*Y
                Y=np.dot(Xt,S)+np.dot(np.dot(Xt,Xo),Y)
            elif DoQR:
                X,R  = linalg.qr(X,mode='economic')
                Xt = X.T
                # Y = Xt*S + (Xt*Xo)*Y
                Y=np.dot(Xt,S)+np.dot(np.dot(Xt,Xo),Y)
            
            elif  Xt == X.T  :#不执行未修改
                
                # Y = Xt*S + (Xt*Xo)*Y
                Y=np.dot(Xt,S)+np.dot(np.dot(Xt,Xo),Y)
                # Y = linsolve(Xt*X,Y,linopts)    ##因为代码没有执行到此，没有修改成功
                Y=linsolve(np.dot(Xt,X),Y,linopts)

            Res = data - partXY(Xt,Y,Ik,Jk,L)  ###partXY是哪个 没有修改
        res =torch.linalg.norm(Res)
        relres = res/datanrm
        ratio = res/res0
        
        
        
        reschg = abs(1-res/res0)
        
        RR[iter] = ratio
        
        if  est_rank >= 1:#进 调用函数，这里为简单全部返回了，有没有更好的办法变量设成全局
            k,est_rank,reschg,tol,rk,rank_max,itr_rank,R,rk_jump, minitr_reduce_rank, maxitr_reduce_rank,rank_min,Zfull,iprint,alf,rk_inc,save_res,X,Y,m,n=rank_estimator_adaptive(k,iter,est_rank,reschg,tol,rk,rank_max,itr_rank,R,rk_jump, minitr_reduce_rank, maxitr_reduce_rank,rank_min,Zfull,iprint,alf,rk_inc,save_res,X,Y,m,n)

        if rk != k:#yes#####
            
            k = rk
            
            
            
            if est_rank ==0:
                alf = 0
                continue
        if ratio >= 1:####yes####
            
            increment = max(0.1*alf, 0.1*increment)
            X = Xo
            Y = Yo
            Res = Res0
            res = res0
            relres = res/datanrm
           
            alf = 0
            if Zfull:
                Z = Zo
                
                
                
        elif ratio > 0.7: ####yes####
            increment = max(increment, 0.25*alf)
            alf = alf + increment
            
        
        if iprint == 1:#无
            
            print('\b\b\b\b\b{}',format(iter))##format()???
        if iprint == 2:#无    #没有执行没有设置精度
            print('it: {} rk: {}, rel. {} r. {} chg: {} alf: {} inc: {}\n',format(iter, k, relres,ratio,reschg,alf0x,increment))
        objv[iter] = relres
        



        #check stopping
        if (((reschg < reschg_tol ) & (itr_rank > minitr_reduce_rank)) or (relres < tol)):
            break
        if Zfull:#进
            
            # Z[Known] = data + alf*Res            
            xx=Z.shape[0]
            yy=Z.shape[1]
            Z=torch.reshape(Z.T,[Z.shape[0]*Z.shape[1],1])
            Z[Known]=data+alf*Res
            Z=torch.reshape(Z,[yy,xx])
            Z=Z.T
            
            
           
            
        else :####nonono####不执行未修改
            # updateSval(S, (alf+1)*Res, L)  #如何从matlab转   
            print('测试注释')
    
    
    
    if iprint == 1 :
        print('\n')
    if data_tran:
        tX = X
        X = Y.T
        Y = tX.T

    
    Out={}     #输出部分数值out字典

    Out['obj'] = objv[0:iter]
    Out['RR'] = RR[0:iter]
    Out['iter'] = iter
    Out['rank'] = rk
    Out['relres'] = relres
    Out['reschg'] = reschg
    
    

    return X,Y,Out

###好像没有调用

def init_inc():
    dr = L/(m+n-k)/k
    increment = .5*log10(m*n) - log10(rk)
    if min(m,n)/k > 1000:
        increment = increment + .25*exp(dr-1)/dr
    

           

###测试


###m,n是cmtx的size,k=1,known 是cmtx上的不等于0序号，data对应的值；opts设成字典


# opts={'maxit':10000,'Zfull':1,'DoQR':1,'print':0,'est_rank':2}#maxit=1e4

# m=1089
# n=507
# k=1
# # data=[]
# Known=[]
# for i in range(1,101):
#     data.append(0.008*i)
#     Known.append(i)
# data=np.array(data)
# Known=np.array(Known)
# data=np.matrix(data)
# # Known=np.matrix(Known)
# data=scio.loadmat(r'C:\Users\汪贵军\Desktop\data.mat')['data']
# Known=scio.loadmat(r'C:\Users\汪贵军\Desktop\Known.mat')['Known'][:,0]
# print(Known)
# # data=data.T
# # Known=Known.T

# U,V,out=lmafit_mc_adp(m,n,k,Known,data,opts)
# print(out)






