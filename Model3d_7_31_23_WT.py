import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tellurium as te
import roadrunner
import re
from scipy.signal import argrelextrema as extrema
from matplotlib import animation, rc
from scipy.interpolate import griddata
from sys import exit
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pPolygon
import matplotlib as mpl
from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
from sys import exit
from scipy.ndimage.morphology import binary_erosion
from scipy.spatial import Voronoi
from skimage import draw
from sklearn.neighbors import KDTree
import pandas as pd
import pytransform3d.coordinates as pc
from mayavi import mlab
#import julia
#from julia import Main
#from julia import Pkg; Pkg.add("GLMakie")
#from julia import GLMakie
#from julia import GeometryBasics
#from julia import LinearAlgebra


#######! if change the number of points, the cutoff, and the cell_total in run_sim() needs to be changed

def get_neib_mx(points, cutoff=100):#100
    '''
    points: (n, 2) numpy array for coordinates
    Return adjacency matrix
    '''
    neib_mx = np.zeros((len(points), len(points)))
    print('len(points):', len(points))
    
    for i, pt1 in enumerate(points):
        #print('i',i)
        #print('pt1',pt1)
        for j in range(i+1, len(points)):
            pt2 = points[j]
            if sum(((pt1 - pt2) ** 2)) < cutoff ** 2:
                    neib_mx[i,j] = 1
                    neib_mx[j,i] = 1
                    
            #print('the neiber pts:', neib_mx)
        
    return neib_mx

def num_pts(points):
    n_pts=len(points)
    return n_pts


def gen_2d_dome(n_layers=7):#6
    #T [1, 6, 12, 18, 24, 30]     #R [0.0, 2.0, 4.0, 6.000000000000001, 8.0, 10.0]
    #TT [1, 6, 12, 18, 24, 30]
    T = [1] + [(x+1)*10 for x in range(n_layers-1)] # T is seperatation of the circle by angles , R is the radius of the circle. These decide the intensity of points
    TT = [1] + [(x+1)*10 for x in range(n_layers-1)]
    R = [0.0] + [(x+1)*0.1 for x in range(n_layers-1)]

    T = [int(np.ceil(x*0.6)) for x in T]
    TT = [int(np.ceil(x*0.6)) for x in TT]
    R = [x*20 for x in R]
    #print('T',T)
    #print('R',R)
    #print('TT',TT)

    def rtpairs(r, nn, n): 
        #rint(len(r))
        #rint(n)
        for i in range(len(r)):
            for k in range(nn[i]):
                for j in range(int(np.ceil(n[i]*np.sin(k*(2 * np.pi / nn[i]))))+1):
                    yield r[i]*1, k*(2 * np.pi / nn[i]),j*(2*np.pi/(1+int(np.ceil(n[i]*np.sin(k*(2 * np.pi / nn[i])))))) # form uniform points, convinient for using cutoff
    points, ppoints = [], []
    for r, tt, t in rtpairs(R, TT, T):
        #print('r',r)
        x, y, z = round(r * np.sin(tt)*np.cos(t),4), round(r *np.sin(tt)*np.sin(t),4), round(r*np.cos(tt),4) #using round to avoid duplicate points
        if z>=-0 and y>=-0:####y>=0 is the original
            if [x,y,z] not in points:
                #rint([x,y,z])
                points.append([x,y,z]) 
    #oints=points+[
    points = np.array([[x*30+0,-y*30+10,z*30+0] for x, y,z in points])
    #print('# of points',range(points))
    #plt.scatter(points[:,0], points[:,1])
    #plt.show()
    #exit()
    return points

def get_l1(points, cutoff=270):# cutoff will change with the layers and others,cutoff=g[:, 2].max()--around radius
        '''
        points: (n, 2) numpy array for coordinates
        Return adjacency matrix
        '''
        l1_pts = []
        for i, pt1 in enumerate(points):
            if i==0: #the center point of the sphere
                print('pt1:',pt1)
                for j in range(i+1, len(points)):
                    pt2 = points[j]
                    if sum(((pt1 - pt2) ** 2)) > cutoff ** 2:
                        l1_pts.append(j)
                        #print('the l1 cells:j',j)
        print('the l1 cells:',l1_pts)  
        print('len of the l1 cells:',len(l1_pts))    
        return l1_pts
    
def get_test(points):

        test_pts = []
        for i, pt1 in enumerate(points):
            if i==0: #the center point of the sphere
                test_pts.append(i)
                for j in range(i+1, len(points)):
                    pt2 = points[j]
                    if sum(((pt1 - pt2) ** 2)) <130**2 and pt2[2]<80:
                        test_pts.append(j)  
        print('test:',test_pts)  
        print('len of test:',len(test_pts))
        return test_pts


def get_lr(points, cutoff=350):# cutoff will change with the layers and others,cutoff=g[:, 2].max()--around radius
        '''
        points: (n, 2) numpy array for coordinates
        Return adjacency matrix
        '''
        lr_pts = []
        for i, pt1 in enumerate(points):
            if i==0: #the center point of the sphere
                print('pt1:',pt1)
                for j in range(i+1, len(points)):
                    pt2 = points[j]
                    if sum(((pt1 - pt2) ** 2)) > cutoff ** 2 and pt2[2] <230 and pt2[1]>-275:
                        lr_pts.append(j)
                        #print('the l1 cells:j',j)
        print('the lr/epfl cells:',lr_pts) 
        print('len of the lr/epfl cells:',len(lr_pts)) 
        return lr_pts 
    
def get_iw(points):

        iw_pts = []
        for i, pt1 in enumerate(points):
            if i==0:
                for j in range(i+1, len(points)):
                    pt2 = points[j]
                    if (pt2[2] - pt1[2]) > 100 and sum(((pt1 - pt2) ** 2)) < 270 ** 2:
                        iw_pts.append(j)  
        print('iw:',iw_pts)                
        return iw_pts
    
def get_mh(points):

        mh_pts = []
        for i, pt1 in enumerate(points):
            if i==0:
                mh_pts.append(i) 
                for j in range(i+1, len(points)):
                    pt2 = points[j]
                    if sum(((pt1 - pt2) ** 2)) < 270 ** 2 and pt2[2]<92:
                        mh_pts.append(j)  
        print('mh/HAM:',mh_pts) 
        print('len of mh/HAM:',len(mh_pts))  
        return mh_pts    

class Model2d:
    def __init__(self):
        self.points = None
        self.neib_mx = None
        #self.l1_pts=None
        self.sp_pts = {}    # Dict of special points
        self.model_str = ''
        self.r = None       # te object
        return None

    def init_2d_dome_model(self, neib_cut=100):# the neib_cut is 100 is because that some neighbered points get so far away
        self.points = gen_2d_dome()
        self.neib_mx = get_neib_mx(self.points, cutoff=neib_cut)
        
    
    def add_l1_pts(self,name=None):
        self.l1_pts=get_l1(self.points,cutoff=270)
        return self.l1_pts
    def add_test_pts(self,name=None):
        self.test_pts=get_test(self.points)
        return self.test_pts
    def add_lr_pts(self,name=None):
        self.lr_pts=get_lr(self.points,cutoff=350)
        return self.lr_pts 
    def add_iw_pts(self,name=None):
        self.iw_pts=get_iw(self.points)
        return self.iw_pts
    def add_mh_pts(self,name=None):
        self.mh_pts=get_mh(self.points)
        return self.mh_pts
        
        
    def num_pts(self):
        self.n_pts=num_pts(self.points)
        return self.n_pts    
    
        
    

    def add_sp_pts(self, ids=[], name=None):
        #rint('ids:',ids)
        self.sp_pts[name] = ids
        #print(self.sp_pts[name])
        return 0

    def add_basal_str(self, model_str=''):
        self.model_str = model_str

    def expand_model(self):
        n = self.points.shape[0] - 1
        model_str = self.model_str
        ms = re.finditer(r"J(\d+)", model_str)
        idj_max = max([int(m.group(1)) for m in ms])
        #print("Max J: ", idj_max)
        ms = re.finditer(r"\bW(\d+)\b", model_str)
        try: 
            idx_max = max([int(m.group(1)) for m in ms])
        except:
            idx_max = 0
        #print("Max X: ", idx_max)
        xnames = ['W'+str((idx_max+i+1)) for i in range(n-idx_max)]
        x_base = xnames[0]
        #print("X names: ", xnames)
        
        reacts = []
        
        model_lines = [x.strip() for x in model_str.split('\n')]
        for li, line in enumerate(model_lines):
            if re.match(r'J', line):
                #print(line)
                reacts.append(line)
                m = re.search(r"W(\d+)?", line)
                if m:
                    x_base = m.group(0)
                    if xnames[0] != x_base:
                        xnames.insert(0,x_base)
    #             model_lines[li] += add_epi_terms(line, xnames, x_base, mx=make_neighbor_mx(n))
        #print(x_base)
        #print(reacts)
        #print(model_lines)
        
        jid = idj_max + 1
        for xname in xnames[1:]:
            for ir, react in enumerate(reacts):
                jname = "J" + str(jid).zfill(2)
                #model_lines.append('\n'+jname+':')
                new0 = re.sub(r"J\d+", jname, react)
                #print(new0)
                new1 = re.sub(x_base, xname, new0)
                #print(new1)
                new2 = re.sub(r'\bC\d+', 'C'+xname[1:], new1)
                new3 = re.sub(r'\bL\d+', 'L'+xname[1:], new2)
                new4 = re.sub(r'\bw\d+', 'w'+xname[1:], new3)
                new5 = re.sub(r'\bc\d+', 'c'+xname[1:], new4)
                new6 = re.sub(r'\bH\d+', 'H'+xname[1:], new5)
                #print(new2)
                #new2 += add_epi_terms(new2, xnames, xname, mx=make_neighbor_mx(n))
                #print(new2)
                model_lines.append(new6)
                jid +=1
            
        for line in model_lines:
            #print(line)
            pass
        #return "\n".join(model_lines)
        self.model_str = "\n".join(model_lines)

    def set_init(self, conds=None):
        reg = r'-> ' + r'(\w+)'
        state_vars = []
        for line in self.model_str.split('\n'):
            m = re.search(reg, line)
            if m:
                state_vars.append(m.group(1))
        self.model_str += '\n\t\n\t//Species Init:\n\t'
        #print(state_vars)
        for v in state_vars:
            for pts_name, cond in conds.items():
                pts = self.sp_pts[pts_name]
                m = re.search(r'([A-Za-z]+)(\d+)', v)
                g, ci = m.group(1), int(m.group(2))
                if ci in pts and g in cond.keys():
                    self.model_str += v + '='+str(cond[g])+'; '
                else:
                    self.model_str += v + '=0.0; '
                    continue
            #self.model_str += v + '=0.0; '
        #print(self.model_str)
        #exit()

    def add_syn2eqn(self, line, var, pts, reg_term=1):
        reg = r'-> ' + var + r'(\d+)'
        m = re.search(reg, line)
        if m and int(m.group(1)) in pts:
            id_str = m.group(1)
            #print(id_str, pts)
            #new_p = 'k0' + var + id_str
            new_p = ''
            #line_new = line + ' + ' + new_p + '*' + re.sub('xxx', id_str, str(reg_term))
            line_new = line + ' + ' + re.sub('xxx', id_str, str(reg_term))
            #print(line_new)
            return line_new, new_p
        else:
            return 0, 0

    def add_syn2model(self, var, pts_name, reg_term=1):
        if pts_name in ['all', 'All', 'ALL', '', None]:
            pts = list(range(self.points.shape[0]))
        else:
            pts = self.sp_pts[pts_name]
        #print(pts)
        #exit()
        ps = []
        for line in self.model_str.split('\n'):
            line_new, new_p = self.add_syn2eqn(line, var, pts, reg_term)
            if line_new:
                if new_p:
                    ps.append(new_p)
                self.model_str = re.sub(re.escape(line), line_new, self.model_str)
                #print(line_new)
                #exit()
        #self.model_str += '\n' + '; '.join([p + ' = 1' for p in ps])

    def add_diffu2eqn(self, model_str, var, x1, x2s, sink_bounds=[]):
        new_p = None
        reg = r'-> ' + var + str(x1) + ';'
        #print(var+str(x1))

        for line in model_str.split('\n'):
            m = re.search(reg, line)
            if m:
                if len(sink_bounds) > 0:
                    if i in sink_bounds:
                        out_cells = str(len(x2s)+1)
                        #out_cells = str(len(x2s)+1)
                        #print(x1, len(x2s), sink_bounds)
                    else:
                        out_cells = str(len(x2s))
                else:
                    out_cells = str(len(x2s))
                new_p = 'D' + var
                line_new = line + ' - ' + new_p + '*' + out_cells + '*' + var + str(x1)
                line_new = line_new + ' + ' + new_p + '*(' + '+'.join([var+str(x) for x in x2s]) + ')'
                #print(line_new)
                model_str = re.sub(re.escape(line), line_new, model_str)
                #print(model_str)
                #exit()
        return model_str, new_p

    def add_diff2model(self, var, sink_bounds=[]):
        model_str = self.model_str
        for i in range(0, len(self.neib_mx)):
            #print(i, np.where(self.neib_mx[i])[0])
            model_str, new_p = self.add_diffu2eqn(model_str, var, i, np.where(self.neib_mx[i])[0], sink_bounds)
        self.model_str = model_str

    def run_sim(self, conds={}, gene_names=[], Ldoses=[], if_perturb=False, cell_comp=False):
        if not self.r:
            r = te.loada(self.model_str)
            self.r = r
        else:
            r = self.r
        r.reset()
        #cp_L_max, cp_gc, cp_gH, cp_KCL1, cp_sc = r.L_max, r.gc, r.gH, r.KCL1, None
        #r.gc, r.gH = gc, gH
        ps_cp = {}
        for p in r.ps():
            ps_cp[p] = r[p]

        for key in conds:
            r[key] = conds[key]

        #print('\n', r.gc, r.gH)
        ms_final = []
        #Ls = np.linspace(0, 15, 15)
        if 'L_max' in conds:
            Ls = [conds['L_max']]
        elif len(Ldoses)>0:
            Ls = Ldoses
        else:
            Ls = [r.L_max]
        glist = r.getFloatingSpeciesIds()
        #print('glist',glist.index('w0'))
        for L_max in Ls:
            r.L_max = L_max
            #r.degc = L_max
            #for var in sels[1:]:
            #for var in r.getFloatingSpeciesIds():
                #if int(var[1:]) in peri_pts:
                    #r.var = np.random.uniform(0, 0.1)
                #else:
                    #r.var = 0
                #print(var, 'c'+var[1:])
                #r[var] = 0
                #r['c'+var[1:]] = np.random.uniform(0, 0.01)
                #r['C'+var[1:]] = np.random.uniform(0, 0.01)
                #if int(var[1:]) in bottom_ini_pts:
                    #r['W'+var[1:]] = np.random.uniform(0.9, 0.99)*0.4*4
                    #r['w'+var[1:]] = np.random.uniform(0.9, 0.99)*0.4*4
                #else:
                    #r['W'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01
                    #r['w'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01
                #r['W'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01
                #r['w'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01

            #r.timeCourseSelections = sels
            #print(r.clearModel())
            if if_perturb == False:
                m = r.simulate (0, 200, 100)
            else:
                m1 = r.simulate(0, 50, 250)
                #for i, g in enumerate(glist):
                    #if 'c' in g or 'C' in g:
                        #r[g] = 0
                        #m1[-1, i+1] = 0
                    #if ('w' in g or'W' in g) and int(g[1:]) in cz_pts:
                        #r[g] = 1
                        #m1[-1, i+1] = 1
                #m2 = r.simulate(50, 100, 250)
                #m = np.vstack((m1,m2))
            ms_final.append(m[-1])
            #r.reset()
            #r.gc, r.gH = gc, gH

        ms_final = np.array(ms_final)
        #print(ms_final.shape)

        idsL = [glist.index(x)+1 for x in glist if 'L' in x]
        idsW = [glist.index(x)+1 for x in glist if 'W' in x]
        idsC = [glist.index(x)+1 for x in glist if 'C' in x]
        idsw = [glist.index(x)+1 for x in glist if 'w' in x]
        idsc = [glist.index(x)+1 for x in glist if 'c' in x]
        idsH = [glist.index(x)+1 for x in glist if 'H' in x]
        #print(m.shape, type(m), m.colnames)
        #exit()
        if 1:
            mGenes, mrGenes = [], []
            for i, ids in enumerate([idsL, idsW, idsC, idsw, idsc, idsH]):
                mrGenes.append(m[:,ids])
                if i == 10:
                    mGene = m[:,ids]
                    mGenes.append(mGene)
                    continue
                mGene = (m[:,ids] - m[:,ids].min()) / (max(m[:,ids].max(),0.3) - m[:,ids].min())
                mGenes.append(mGene)
                #print('Total', gene_names[i], "%.2f"%m[-1,ids].sum())
            mGenes, mrGenes = np.array(mGenes), np.array(mrGenes)
            mL, mW, mC, mw, mc, mH = mGenes
            mT = m[:,0]
            
        
        
        if len(Ldoses)>0:
            fig, axs = plt.subplots(figsize=[6, 2.5], ncols=2)
            fig.subplots_adjust(wspace=0.7, bottom=0.2, right=0.7)
            for i, ids in enumerate([idsL, idsW, idsC, idsw, idsc, idsH]):
                if gene_names[i] in ['CLV3 mRNA', 'WUS mRNA']:
                    #print(ms_final.shape, len(idsL))
                    #exit()
                    gene_total = ms_final[:,ids].sum(axis=1)
                    cell_total = (ms_final[:,ids] > 0.5).sum(axis=1) / 326###################################################
                    #cell_total = (ms_final[:,ids][:,wus_peri_pts] > 0.5).sum(axis=1) / len(wus_peri_pts)
                    if gene_names[i] == 'CLV3 mRNA':
                        c = 'purple'
                        xc = gene_total
                    if gene_names[i] == 'WUS mRNA':
                        c = 'orange'
                        xw = gene_total
                    axs[0].plot(Ls, gene_total, lw=3, label=gene_names[i], c=c)
                    axs[1].plot(Ls, cell_total, lw=3, label=gene_names[i], c=c)
            axs[0].plot(Ls, xw/xc, lw=3, c='k',zorder=-10, label='WUS:CLV3 Ratio')
            rt = xw/xc
            axs[0].set_title("%.2f %.2f %.2f %.2f"%(rt.min(), rt.max(), rt.std(), rt.std()/rt.mean()))
            for ax in axs:
                #ax.legend()
                ax.set_xlabel('EPFL signal', size=10)
                #ax.set_xlim([-0.3, 15.3])
                ax.set_xticklabels(['', '50%', '100%', '150%'])
            axs[0].set_ylabel('Total amount in SAM', size=10)  
            #axs[0].set_ylim([-0.3, 35])
            axs[1].set_ylabel('Fraction of high\nexpressing cells in SAM', size=10)  
            axs[1].legend(bbox_to_anchor=[1.05, 0.1, 0.1, 0.6], prop={'size':10})
            axs[1].set_yticks([0.15, 0.2, 0.25])
            #axs[1].set_ylim([-0.05, 1.05])
            plt.show()
            r.reset()
            return fig
            #exit()

        if cell_comp:
            fig, ax = plt.subplots(figsize=(3,3))
            ax = fig.add_subplot(projection='3d')
            fig.subplots_adjust(left=0.22, bottom=0.22)
            ax.scatter(ms_final[:,idsw], ms_final[:,idsc], alpha=0.4, fc='k')
            ax.set_xlabel('WUS mRNA', size=12)
            ax.set_ylabel('CLV3 mRNA', size=12)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            r.reset()
            return fig

        if 0:
            fig, ax = plt.subplots(figsize=[20, 3])
            ax = fig.add_subplot(projection='3d')

            #for i in range(1, m.shape[1])[]:
            for i in idsC:
                ax.plot(m.T[0], m.T[i], alpha=0.5)
            ax.set_xlabel('Time', size=12)
            ax.set_ylabel('Abundance', size=12)


        for p in self.r.ps():
            self.r[p] = ps_cp[p]
        r.reset()
        #print(r['kc'])
        #exit()
        #r.L_max, r.gc, r.gH, r.KCL1, r_sc = cp_L_max, cp_gc, cp_gH, cp_KCL1, cp_sc
        #print(mGenes.shape)
        #exit()
        #mGenes = pd.DataFrame(mGenes, columns=['L', 'W', 'C', 'w', 'c', 'H'])
        return mGenes, mrGenes, m[:,0]

    def plot_cbar(self, cax, cmap='inferno'):
        
        cax.imshow(np.vstack((np.linspace(0, 1, 256), np.linspace(0, 1, 256))).T, cmap=cmap, 
                   aspect='auto', origin='lower')
        cax.yaxis.set_ticks_position('right')
        cax.set_xticks([])
        cax.set_yticks(np.linspace(0, 256, 6))
        cax.set_yticklabels(["%.1f"%(i/256) for i in np.linspace(0, 256, 6)])
        cax.set_ylabel('Scaled abundance')

    def plot_tps(self, data, tps, gtype):
        
        
        n_tpts = 5
        fig, axs = plt.subplots(ncols=len(data), nrows=n_tpts, figsize=[7,4])
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        ax = fig.add_subplot(projection='3d')
        cax = fig.add_axes([0.93,0.7,0.01,0.15])
        cmap = plt.get_cmap('jet')
        self.plot_cbar(cax, cmap=cmap)
        scatters = []
        tpts_ids = [int(x) for x in np.linspace(0, data[0].shape[0]-1, n_tpts)]
        gene_order = [0,3,1,4,2,5]
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                ax = axs[i,j]
                ax.set_aspect('equal')
                ax.set_facecolor('w')
                ax.axis('off')
                #for ci, poly in enumerate(new_vertices):
                    #f = ax.fill(*zip(*poly), alpha=0.99, edgecolor='k',
                            #facecolor=cmap(data[gene_order][j][tpts_ids[i], ci]))
                if j == 10:
                    vmax = (self.r.k0WL+self.r.k0WC)*self.r.k0WW
                else:
                    vmax = max(data[gene_order][j].max(),1)
                ax.scatter(self.points[:,0], self.points[:,1],self.points[:,2], cmap=cmap, \
                    c=data[gene_order][j][tpts_ids[i], :], norm=mpl.colors.Normalize(vmin=0, vmax=vmax))

        for j, name in zip(range(axs.shape[1]), 
                           ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']):
            axs[0,j].set_title(name, size=10)
            #axs[0,j].set_title(name+" %.2f"%mrGenes[gene_order][j,:,:].max())
            #print(name, "%.2f"%data[gene_order][j,:,:].max())
        for i in range(axs.shape[0]):
            #axs[i,0].text(0, 480, 'Time '+str(i), size=10)
            axs[i,0].text(-180, 400, 'Time '+ str(int(tps[tpts_ids][i])), size=10)

        #print(r.gc, r.gH, r.L_max, '\n')
        #fig.suptitle("L_max: %.2f  gc: %.2f  gH: %.2f " % (r.L_max, r.gc, r.gH))
        fig.suptitle(gtype)

    
        
    def plot_all_gts(sam, data_all,  gtypes,gid=0): # here gid is col of the original picture in 2d, so its ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']
        n_tpts = len(data_all)  #6
        cmap = plt.get_cmap('jet')
        gene_order = [0,3,1,4,2,5]
        j = gid
        inten_type= ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']
        
        
        print('gid',gid)
        #fig, axs = plt.subplots(ncols=1, nrows=n_tpts, figsize=(2,5))
        fig = plt.figure(figsize=plt.figaspect(6))
        for i in range(n_tpts): # here i means row the the whole plot(wt and other mutants)
            dataij = data_all[i][1][gene_order][j][-1,:]
            vmax = 1
            vmin = 0
            ax = fig.add_subplot(n_tpts, 1, i+1, projection='3d')          
            ax.scatter(sam.points[:,0], sam.points[:,1],sam.points[:,2], cmap=cmap, \
                c=dataij, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            ax.set_title('ge_type: '+ gtypes[i]+' \n'+'Type: '+str(inten_type[j]))
            #fig=GLMakie.meshscatter(sam.points[:,0], sam.points[:,1],sam.points[:,2], markersize = 0.3,color=dataij)
            #GLMakie.save("landscape3.png", fig, pt_per_unit=1)
        fig.subplots_adjust(hspace=0.3)
        plt.savefig("1.png")
     
    def plot_all_gts_julia(sam, data_all,  gtypes,gid=0): # here gid is col of the original picture in 2d, so its ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']
        n_tpts = len(data_all)  #6
        cmap = plt.get_cmap('jet')
        gene_order = [0,3,1,4,2,5]
        j = gid
        inten_type= ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']
        
        
        print('gid',gid)
        #fig, axs = plt.subplots(ncols=1, nrows=n_tpts, figsize=(2,5))
     
        for i in range(n_tpts): # here i means row the the whole plot(wt and other mutants)
            i=2
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            dataij = data_all[i][1][gene_order][j][-1,:]
            vmax = 1
            vmin = 0         
            ax.scatter(sam.points[:,0], sam.points[:,1],sam.points[:,2], cmap=cmap, \
                c=dataij, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            ax.set_title('ge_type: '+ gtypes[i]+' \n'+'Type: '+str(inten_type[j]))
            fig1=GLMakie.meshscatter(sam.points[:,0],sam.points[:,1], sam.points[:,2], markersize = 30,color=dataij,rotations=np.pi)
            GLMakie.save("landscape4.png", fig1, pt_per_unit=1)#this can't rotate.
     
    def plot_all_gts_mlab(sam, data_all,  gtypes): #(...,gid=0) here gid is col of the original picture in 2d, so its ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']
        n_tpts = len(data_all)  #6
        cmap = plt.get_cmap('jet')
        gene_order = [0,3,1,4,2,5]
        jj = [0,1,2,3,4,5]
        inten_type= ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']
        #mlab.colorbar()
        
        
        for i in range(n_tpts): # here i means row the the whole plot(wt and other mutants)
            for j in jj:
                #fig = plt.figure()
                fig = mlab.figure()
                #ax = plt.axes(projection='3d')
                dataij = data_all[i][1][gene_order][j][-1,:]
                vmax = 1
                vmin = 0 
                '''
                ax.scatter(sam.points[:,0], sam.points[:,1],sam.points[:,2], cmap=cmap, \
                    c=dataij, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
                ax.set_title('ge_type: '+ gtypes[i]+' \n'+'Type: '+str(inten_type[j]))
                '''
                colors = dataij

                nodes = mlab.points3d(sam.points[:,0], sam.points[:,1],sam.points[:,2])
   
                nodes.glyph.scale_mode = 'scale_by_vector'
                nodes.mlab_source.dataset.point_data.vectors = np.tile( 2*np.ones((5000,)), (3,1))
                nodes.mlab_source.dataset.point_data.scalars = colors
                nodes.actor.actor.orientation = [25,25,-90]
                 #[20,28,-90]   [30,29,-90]
                   
                
                # Set the background color to white
                fig.scene.background = (1, 1, 1)
                #
                
                 
               
                
                mlab.savefig(f'output_figure_{i}_{j}.png', magnification=10, size=(3600, 2400))
                mlab.show()
        
        
                       
        

    def plot_points(self, ax=None, s=10, no_num=False):
        pts = self.points
        #print(pts)
        g = gen_2d_dome()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(self.points[:, 0].min(), self.points[:, 0].max())
        ax.set_ylim(self.points[:, 1].min(), self.points[:, 1].max())
        ax.set_zlim(self.points[:, 2].min(), self.points[:, 2].max())
        

        #ax = make_3d_axis(ax_s=300, unit="m", n_ticks=7) ###
        #ax.view_init(0, 90)
        if ax == None:
            fig, ax = plt.subplots()
        ax.scatter(pts[:,0], pts[:,1],pts[:,2], s=s, color='black')
        
        '''
        if no_num == False:
            for i, pt in enumerate(pts):
                ax.text(pt[0],pt[1],pt[2], str(i), zorder=10)
        #plt.show()
        '''
        return ax
    
    def plot_sp_pts(self, name=None):
        pts = self.points
        
        #fig = plt.figure()
        #ax = plt.figure().add_subplot(projection='3d')
        '''
        if name:
            ids = self.sp_pts[name]
            ax = self.plot_points(s=1)
            ax.scatter(pts[ids,0], pts[ids,1], pts[ids,2],s=100, color='black')
            ax.set_title(name, color='black', size=12)
            print()
        else:
            for name in self.sp_pts.keys():
                ids = self.sp_pts[name]
                ax = self.plot_points(s=1)
                ax.scatter(pts[ids,0], pts[ids,1],pts[ids,2], s=100, color='black')
                ax.set_title(name, color='black', size=17)
        ax.view_init(90, 90)
        #plt.show()
        '''
        if name:
            ids = self.sp_pts[name]
            ax = self.plot_points(s=1)
            ax.scatter(pts[ids,0], pts[ids,1], pts[ids,2],s=100, color='black')
            ax.set_title(name, color='black', size=12)
            print()
        else:
            for name in self.sp_pts.keys():
                ids = self.sp_pts[name]
                ax = self.plot_points(s=1)
                ax.scatter(pts[ids,0], pts[ids,1],pts[ids,2], s=100, color='black')
                ax.set_title(name, color='black', size=17)
        ax.view_init(4, 76)      
        plt.savefig('sp_figure.png', dpi=600)
        plt.show()
        
        '''
        if name:
            ids = self.sp_pts[name]
            ax = self.plot_points(s=1)
            ax.scatter(pts[ids,0], pts[ids,1], pts[ids,2],s=100, color='black')
            ax.set_title(name, color='black', size=12)
            print()
        else:
            for name in self.sp_pts.keys():
                ids = self.sp_pts[name]
                ax = self.plot_points(s=1)
                ax.scatter(pts[ids,0], pts[ids,1],pts[ids,2], s=100, color='black')
                ax.set_title(name, color='black', size=17)
       
        
        plt.savefig('sp_figure.jpg', dpi=600)
        plt.close(fig)
        plt.show()
        '''


    def plot_neib(self):
        dim = 0
        for i in range(10):
            if i*i > len(self.points):
                dim = i
                break
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig, axs = plt.subplots(ncols=i, nrows=i, figsize=(12, 8))
        axs = axs.flatten()
        for i in range(self.neib_mx.shape[0]):
            ax = axs[i]
            neib_pts = self.points[np.where(self.neib_mx[i])[0]]
            ax = self.plot_points(ax=ax, no_num=True)
            ax.scatter(neib_pts[:,0], neib_pts[:,1],s=10)
            ax.scatter(self.points[i,0], self.points[i,1], s=10)
            ax.set_xticks([])
            ax.set_yticks([])
        #fig.tight_layout()

