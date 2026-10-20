import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model import defaults, simulate, simulate_dimethicone_washes, SOLVENTS

st.set_page_config(page_title='Dual ANFD First-Principles Model',layout='wide')
st.title('Dual ANFD Heat, Mass Transfer & Discrete Dimethicone Wash Model')
st.caption('Rosenmund fixed vertical cake | OMCA tilt-dependent bed | discrete wash-by-wash dimethicone balance')
kind=st.sidebar.selectbox('Equipment',['Rosenmund','OMCA'])
eq,prod,rec=defaults(kind)

st.sidebar.header('Geometry and operation')
eq.filter_area_m2=st.sidebar.number_input('Filter area (m²)',0.001,2.0,float(eq.filter_area_m2),0.001)
eq.vessel_diameter_m=st.sidebar.number_input('Internal diameter (m)',0.05,3.0,float(eq.vessel_diameter_m),0.005)
eq.vessel_length_m=st.sidebar.number_input('Usable length/height (m)',0.05,5.0,float(eq.vessel_length_m),0.01)
eq.headspace_m3=st.sidebar.number_input('Headspace (m³)',0.001,5.0,float(eq.headspace_m3),0.001)
eq.rpm=st.sidebar.slider('Drying agitator RPM',0,120,int(eq.rpm))
eq.agitation_duty=st.sidebar.slider('Drying agitation duty',0.0,1.0,float(eq.agitation_duty),0.05)
if kind=='OMCA': eq.tilt_deg=st.sidebar.slider('Drying tilt angle (°)',-90,90,int(eq.tilt_deg))

st.sidebar.header('Heat transfer')
eq.base_UA_WK=st.sidebar.number_input('Base UA (W/K)',0.0,1000.0,float(eq.base_UA_WK),0.5)
eq.wall_UA_WK=st.sidebar.number_input('Wall UA (W/K)',0.0,1000.0,float(eq.wall_UA_WK),0.5)
eq.agitator_UA_WK=st.sidebar.number_input('Agitator UA (W/K)',0.0,1000.0,float(eq.agitator_UA_WK),0.5)

st.sidebar.header('Product')
prod.dry_mass_kg=st.sidebar.number_input('Dry solids (kg)',0.01,1000.0,float(prod.dry_mass_kg),0.1)
prod.wet_bulk_density_kgm3=st.sidebar.number_input('Wet bulk density (kg/m³)',50.0,2000.0,float(prod.wet_bulk_density_kgm3),10.0)
prod.dry_Tg_C=st.sidebar.number_input('Dry PLGA Tg (°C)',0.0,100.0,float(prod.dry_Tg_C),1.0)
for s in SOLVENTS: prod.initial_solvent_kg[s]=st.sidebar.number_input(f'Initial {s} (kg)',0.0,100.0,float(prod.initial_solvent_kg[s]),0.005)
prod.initial_dimethicone_kg=st.sidebar.number_input('Initial dimethicone/PDMS (kg)',0.0,100.0,float(prod.initial_dimethicone_kg),0.001)
prod.extractable_dimethicone_fraction=st.sidebar.slider('Initially extractable dimethicone fraction',0.0,1.0,float(prod.extractable_dimethicone_fraction),0.05)

st.sidebar.header('Drying recipe')
rec.total_h=st.sidebar.number_input('Drying simulation duration (h)',1.0,200.0,float(rec.total_h),1.0)
rec.pressure_mbar=st.sidebar.number_input('Absolute pressure (mbar)',1.0,1000.0,float(rec.pressure_mbar),5.0)
eq.condenser_temp_C=st.sidebar.number_input('Condenser temperature (°C)',-80.0,30.0,float(eq.condenser_temp_C),1.0)
profile_text=st.sidebar.text_area('Jacket profile: hour,temp per line','\n'.join(f'{a},{b}' for a,b in rec.jacket_profile),height=130)
try: rec.jacket_profile=[tuple(map(float,x.split(','))) for x in profile_text.splitlines() if x.strip()]
except ValueError: st.sidebar.error('Use hour,temp format')

st.subheader('Discrete dimethicone wash cycles')
st.info('K is the liquid-to-solid partition coefficient in L/kg. kLa controls approach to equilibrium. Drain fraction is recovered liquid divided by liquid present at cycle end.')
default_cycles=pd.DataFrame([
 {'name':'Wash 1','volume_L':3.0,'retained_L':0.30,'contact_min':30.0,'rpm':30.0,'tilt_deg':0.0,'temp_C':25.0,'K_L_per_kg':2.0,'kLa_1_per_h':0.35,'drain_fraction':0.95},
 {'name':'Wash 2','volume_L':3.0,'retained_L':0.30,'contact_min':30.0,'rpm':30.0,'tilt_deg':0.0,'temp_C':25.0,'K_L_per_kg':2.0,'kLa_1_per_h':0.35,'drain_fraction':0.95},
 {'name':'Wash 3','volume_L':3.0,'retained_L':0.30,'contact_min':30.0,'rpm':30.0,'tilt_deg':0.0,'temp_C':25.0,'K_L_per_kg':2.0,'kLa_1_per_h':0.35,'drain_fraction':0.95},
])
cycles_df=st.data_editor(default_cycles,num_rows='dynamic',use_container_width=True,hide_index=True,key='cycles')
cycles=cycles_df.to_dict('records')
wash_df,wash_state=simulate_dimethicone_washes(eq,prod,cycles)

df,meta=simulate(eq,prod,rec)
cols=st.columns(6)
cols[0].metric('Cake/bed depth',f"{meta['geometry']['cake_depth_m']*1000:.1f} mm")
cols[1].metric('Minimum Tg margin',f"{df.Tg_margin_C.min():.1f} °C")
cols[2].metric('Final DCM',f"{df.DCM_ppm.iloc[-1]:,.0f} ppm")
cols[3].metric('Final ethanol',f"{df.Ethanol_ppm.iloc[-1]:,.0f} ppm")
cols[4].metric('Dimethicone extracted',f"{(wash_df.extraction_percent.iloc[-1] if len(wash_df) else 0):.1f}%")
cols[5].metric('Retained dimethicone',f"{(wash_df.dimethicone_retained_ppm.iloc[-1] if len(wash_df) else prod.initial_dimethicone_kg/prod.dry_mass_kg*1e6):,.0f} ppm")

if len(wash_df):
    f=go.Figure()
    f.add_bar(x=wash_df.name,y=wash_df.dimethicone_recovered_this_cycle_g,name='Recovered per cycle (g)')
    f.add_scatter(x=wash_df.name,y=wash_df.dimethicone_retained_ppm,name='Retained ppm',yaxis='y2',mode='lines+markers')
    f.update_layout(title='Discrete dimethicone extraction by wash',yaxis=dict(title='Recovered per cycle (g)'),yaxis2=dict(title='Retained dimethicone (ppm)',overlaying='y',side='right'),hovermode='x unified')
    st.plotly_chart(f,use_container_width=True)
    st.dataframe(wash_df,use_container_width=True)
    st.download_button('Download wash-cycle balance CSV',wash_df.to_csv(index=False).encode(),'dimethicone_wash_cycles.csv','text/csv')

f=go.Figure()
for y,n in [('jacket_C','Jacket'),('product_C','Product'),('Tg_C','Dynamic Tg')]: f.add_scatter(x=df.time_h,y=df[y],name=n)
f.update_layout(title='Drying temperature and Tg',xaxis_title='Time (h)',yaxis_title='Temperature (°C)',hovermode='x unified')
st.plotly_chart(f,use_container_width=True)
f=go.Figure()
for s in SOLVENTS: f.add_scatter(x=df.time_h,y=df[f'{s}_ppm'],name=s)
f.update_layout(title='Volatile residual solvents during drying',xaxis_title='Time (h)',yaxis_title='ppm on dry solids',yaxis_type='log',hovermode='x unified')
st.plotly_chart(f,use_container_width=True)

st.subheader('Checks and limitations')
if len(wash_df) and abs(wash_df.mass_balance_percent.iloc[-1]-100)>0.1: st.error('Dimethicone mass balance does not close.')
else: st.success('Dimethicone mass balance closes at 100% within numerical precision.')
if df.Tg_margin_C.min()<5: st.warning('Tg safety margin falls below 5 °C.')
if kind=='OMCA': st.warning('OMCA GA dimensions remain editable placeholders until approved dimensions are entered.')
st.caption('Calibrate K and kLa from dimethicone assays in each filtrate/wash fraction and final solids. Default values are demonstrative, not validated.')
st.download_button('Download drying simulation CSV',df.to_csv(index=False).encode(),'anfd_drying_simulation.csv','text/csv')
