from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd

SOLVENTS = {
    'DCM': {'MW':0.08493,'Hvap':330000.,'A':6.95465,'B':1170.966,'C':226.232,'Tg':148.0},
    'Ethanol': {'MW':0.04607,'Hvap':841000.,'A':8.20417,'B':1642.89,'C':230.3,'Tg':97.0},
    'Heptane': {'MW':0.10021,'Hvap':317000.,'A':6.89385,'B':1264.37,'C':216.432,'Tg':100.0},
}

@dataclass
class Equipment:
    name: str
    filter_area_m2: float
    vessel_diameter_m: float
    vessel_length_m: float
    headspace_m3: float
    base_UA_WK: float
    wall_UA_WK: float
    agitator_UA_WK: float
    pump_speed_m3s: float
    condenser_temp_C: float
    tilt_deg: float = 0.0
    rpm: float = 30.0
    agitation_duty: float = 0.5
    reverse_blow: bool = False

@dataclass
class Product:
    dry_mass_kg: float
    wet_bulk_density_kgm3: float
    porosity: float
    cp_JkgK: float
    k_WmK: float
    dry_Tg_C: float
    particle_d_um: tuple
    particle_mass_frac: tuple
    initial_solvent_kg: dict

@dataclass
class Recipe:
    total_h: float
    dt_s: float
    pressure_mbar: float
    nitrogen_slpm: float
    jacket_profile: list  # [(hour, C), ...]
    agitation_interval_min: float = 30.0


def psat_Pa(solvent, T_C):
    p=SOLVENTS[solvent]
    # Antoine, mmHg
    return 133.322 * 10**(p['A']-p['B']/(p['C']+T_C))

def interp_profile(profile, hour):
    xs=np.array([p[0] for p in profile],float); ys=np.array([p[1] for p in profile],float)
    return float(np.interp(hour,xs,ys))

def cake_geometry(eq, prod):
    wet_vol=prod.dry_mass_kg/max(prod.wet_bulk_density_kgm3,1e-9)
    if 'OMCA' in eq.name.upper():
        # partially filled horizontal cylinder; equivalent mean depth and contact factors
        cross=max(wet_vol/max(eq.vessel_length_m,1e-6),1e-8)
        R=eq.vessel_diameter_m/2
        # numerical inversion of circular-segment area
        hs=np.linspace(1e-5,2*R-1e-5,1000)
        seg=R*R*np.arccos(np.clip((R-hs)/R,-1,1))-(R-hs)*np.sqrt(np.maximum(2*R*hs-hs*hs,0))
        h=float(hs[np.argmin(np.abs(seg-cross))])
        theta=math.radians(eq.tilt_deg)
        depth=h*(1+0.35*abs(math.sin(theta)))
        free_width=2*math.sqrt(max(2*R*h-h*h,1e-9))
        free_area=free_width*eq.vessel_length_m
        wall_contact=(2*R*math.acos(np.clip((R-h)/R,-1,1)))*eq.vessel_length_m
        filter_cover=float(np.clip(wet_vol/(eq.filter_area_m2*max(depth,1e-6)),0.15,1.0))
        return depth, free_area, wall_contact, filter_cover
    depth=wet_vol/max(eq.filter_area_m2,1e-9)
    free_area=eq.filter_area_m2
    wall_contact=math.pi*eq.vessel_diameter_m*depth
    return depth, free_area, wall_contact, 1.0

def dynamic_tg_C(prod, solvent_mass):
    mp=max(prod.dry_mass_kg,1e-9); total=mp+sum(max(v,0) for v in solvent_mass.values())
    wp=mp/total
    num=wp*(prod.dry_Tg_C+273.15); den=wp
    k={'DCM':2.5,'Ethanol':2.0,'Heptane':1.3}
    for s,m in solvent_mass.items():
        wi=max(m,0)/total; num += k[s]*wi*SOLVENTS[s]['Tg']; den += k[s]*wi
    return num/max(den,1e-9)-273.15

def simulate(eq:Equipment, prod:Product, rec:Recipe, diffusivity=None):
    diffusivity=diffusivity or {'DCM':2.0e-14,'Ethanol':8.0e-15,'Heptane':5.0e-15}
    depth, free_area, wall_contact, filter_cover=cake_geometry(eq,prod)
    dt=rec.dt_s; n=int(rec.total_h*3600/dt)+1
    bins=np.array(prod.particle_d_um)*1e-6
    wf=np.array(prod.particle_mass_frac,float); wf=wf/wf.sum()
    # solvent mass by size bin
    m={s:prod.initial_solvent_kg.get(s,0)*wf.copy() for s in SOLVENTS}
    T=25.0; head={s:0.0 for s in SOLVENTS}; condensate={s:0.0 for s in SOLVENTS}
    rows=[]
    gas_R=8.314
    for q in range(n):
        t=q*dt; hour=t/3600; Tj=interp_profile(rec.jacket_profile,hour)
        ag_on=((t/60)%max(rec.agitation_interval_min,1e-6)) < rec.agitation_interval_min*eq.agitation_duty
        ag_factor=(1+0.012*eq.rpm) if ag_on else 1.0
        if 'OMCA' in eq.name.upper():
            theta=math.radians(eq.tilt_deg)
            redistribution=1+0.6*abs(math.sin(theta))*ag_factor
            contact_factor=0.65+0.35*filter_cover
        else:
            redistribution=ag_factor; contact_factor=1.0
        # vapor headspace partial pressures from moles
        Ppart={s:head[s]*gas_R*(T+273.15)/max(eq.headspace_m3,1e-8) for s in SOLVENTS}
        rates={}; total_latent=0.0
        for s in SOLVENTS:
            # intraparticle diffusion, first eigenmode for sphere; external/vacuum limitation added as resistances
            D=diffusivity[s]*math.exp(3500/8.314*(1/298.15-1/(T+273.15)))
            kint=15*D/np.maximum(bins/2,1e-9)**2
            # activity declines near endpoint; equilibrium pressure suppresses flux
            mtot=max(m[s].sum(),1e-15)
            activity=np.clip(mtot/max(prod.initial_solvent_kg.get(s,1e-12),1e-12),0,1)**0.55
            driving=max(0.0,1-Ppart[s]/max(activity*psat_Pa(s,T),1.0))
            kcake=(2e-5/max(depth,0.005)**2)*redistribution
            kvac=max(eq.pump_speed_m3s/max(eq.headspace_m3,1e-9),1e-8)
            keff=1/(1/np.maximum(kint,1e-12)+1/max(kcake,1e-12)+1/kvac)
            dm=keff*m[s]*driving*dt
            dm=np.minimum(dm,m[s]); m[s]-=dm
            evap=float(dm.sum()/dt)
            # headspace + condenser/vacuum removal
            head[s]+=float(dm.sum())/SOLVENTS[s]['MW']
            p_sat_cond=psat_Pa(s,eq.condenser_temp_C)
            capture=np.clip(1-p_sat_cond/max(Ppart[s]+1,1),0,0.98)
            removal=min(head[s], head[s]*(eq.pump_speed_m3s/max(eq.headspace_m3,1e-9))*dt)
            cond=removal*capture; head[s]-=removal; condensate[s]+=cond*SOLVENTS[s]['MW']
            rates[s]=evap; total_latent += evap*SOLVENTS[s]['Hvap']
        # orientation-dependent available UA
        UA=(eq.base_UA_WK*filter_cover*contact_factor + eq.wall_UA_WK*min(1.0,wall_contact/(math.pi*eq.vessel_diameter_m*max(depth,1e-6))) + (eq.agitator_UA_WK if ag_on else 0))
        qheat=UA*(Tj-T)
        Ctot=prod.dry_mass_kg*prod.cp_JkgK + sum(m[s].sum()*1800 for s in SOLVENTS)
        T += (qheat-total_latent)*dt/max(Ctot,1.0)
        T=float(np.clip(T,-20,Tj+2))
        smass={s:float(m[s].sum()) for s in SOLVENTS}; tg=dynamic_tg_C(prod,smass)
        row={'time_h':hour,'jacket_C':Tj,'product_C':T,'Tg_C':tg,'Tg_margin_C':tg-T,'cake_depth_mm':depth*1000,
             'free_area_m2':free_area,'filter_coverage':filter_cover,'agitation_on':ag_on,'heat_W':qheat,'latent_W':total_latent}
        for s in SOLVENTS:
            row[f'{s}_ppm']=smass[s]/prod.dry_mass_kg*1e6
            row[f'{s}_rate_gph']=rates[s]*3.6e6
            row[f'{s}_condensed_g']=condensate[s]*1000
        rows.append(row)
    df=pd.DataFrame(rows)
    return df, {'equipment':asdict(eq),'product':asdict(prod),'recipe':asdict(rec),'geometry':{'cake_depth_m':depth,'free_area_m2':free_area,'wall_contact_m2':wall_contact,'filter_coverage':filter_cover}}


def defaults(kind='Rosenmund'):
    if kind=='Rosenmund':
        eq=Equipment('Rosenmund vertical ANFD',0.03,0.195,0.35,0.025,5.0,9.0,2.0,0.002,5.0,0,30,0.5)
    else:
        # OMCA dimensions are editable placeholders pending approved dimensional schedule
        eq=Equipment('OMCA tilting ANFD',0.05,0.32,0.55,0.040,6.0,12.0,2.5,0.002,5.0,25,30,0.5)
    prod=Product(1.8,450,0.45,1400,0.12,48,(20,60,120),(0.2,0.6,0.2),{'DCM':0.045,'Ethanol':0.030,'Heptane':0.120})
    rec=Recipe(80,60,80,1.0,[(0,25),(8,30),(20,35),(36,40),(56,43),(76,43),(80,25)],30)
    return eq,prod,rec

