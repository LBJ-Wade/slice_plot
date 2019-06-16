def AHF_halo(path):
    f = open(path)
    data = np.loadtxt(f)
    f.close()
    return np.transpose(data)

def AHF_profile(path):
    f = open(path)
    data = np.loadtxt(f)
    profile = np.transpose(data)
    header=0
    number=np.argwhere(profile[3]<200)
    f.close()
    return header,profile,number

def ESD(snapshot,halocat):
    p,v,m,t,z=PyGadget(snapshot)
    x=np.transpose(p)[0]
    y=np.transpose(p)[1]
    z=np.transpose(p)[2]

    halo=AHF_halo(halocat)
    mass = halo[3]
    flag = halo[1]
    hx = halo[5]
    hy = halo[6]
    hz = halo[7]
    nr=[]
    nr_edditon=[]
    rand_edditon = np.random.normal(loc=np.log10(mass),scale=0.3)
    for i in np.arange(0,1771,1,dtype=int):
        nx,ny,nnr=cut(hx[i],hy[i],hz[i],10000,5000,x,y,z)
        nr=np.append(nr,nnr)
        j=np.argmax(rand_edditon)
        nx_edditon,ny_edditon,nnr_edditon=cut(hx[j],hy[j],hz[j],10000,5000,x,y,z)
        nr_edditon=np.append(nr_edditon,nnr_edditon)
        rand_edditon[j]=0
    surfd = np.histogram(np.log10(nr),bins=50)
    sigma_sr = np.cumsum(surfd[0])*m/(np.pi*(10**(2.0*surfd[1][1:])))
    sigma_r = surfd[0]*m/(np.pi*(10.0**(2.0*surfd[1][1:])-10.0**(2.0*surfd[1][:-1])))
    esd = sigma_sr-sigma_r
    esd = esd*unit_trans/1771

    surfd_edditon = np.histogram(np.log10(nr_edditon),bins=50)
    sigma_sr_edditon = np.cumsum(surfd_edditon[0])*m/(np.pi*(10**(2.0*surfd_edditon[1][1:])))
    sigma_r_edditon = surfd_edditon[0]*m/(np.pi*(10.0**(2.0*surfd_edditon[1][1:])-10.0**(2.0*surfd_edditon[1][:-1])))
    esd_edditon = sigma_sr_edditon-sigma_r_edditon
    esd_edditon = esd_edditon*unit_trans/1771
    return 10.0**surfd[1],esd,10.0**surfd_edditon[1],esd_edditon

unit_trans = 0.68*10**10/1000**2

def periodic_d(a,b,boxsize):
    d = np.min([np.abs(a-b),np.abs(a+boxsize-b)],axis=0)
    d = np.min([d,np.abs(a-boxsize-b)],axis=0)
    return d

def cut(halox,haloy,haloz,cutr,cutz,px,py,pz):
    cut1 = periodic_d(haloz,pz,400000)<=cutz
    dx = periodic_d(halox,px[cut1],400000)
    dy = periodic_d(haloy,py[cut1],400000)
    dr = np.sqrt(dx**2.+dy**2.)
    cut2 = dr<=cutr
    samplex = dx[cut2]
    sampley = dy[cut2]
    sampler =  dr[cut2]
    return samplex,sampley,sampler
