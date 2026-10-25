"""Reproduce route EDA from supplementary labels and official observed input tracks.

python build_route_analysis.py --tracking-root <official train directory>
No network, event-time assumptions, output frames, or inherited teammate labels.
"""
from pathlib import Path
import argparse, base64, json, html, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

ROUTES = ['HITCH','OUT','FLAT','CROSS','GO','IN','SLANT','POST','ANGLE','CORNER','SCREEN','WHEEL']
COLORS = ['#245e86','#cf7437','#47876e']
HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.16,'savefig.dpi':160})

def save(fig, name):
    fig.text(.01,-.015,'Source: supplementary route labels + official 2023 input tracking, weeks 1–18 | Targeted receivers only',fontsize=8,color='#555555')
    fig.savefig(OUT / (name+'.png'), bbox_inches='tight')
    plt.close(fig)

def axis(ax, title):
    ax.set(title=title, xlabel='Lateral displacement (yd; + toward initial sideline)', ylabel='Downfield displacement from window start (yd)')
    ax.axhline(0,color='#888888',lw=.7,ls='--')
    ax.scatter([0],[0],s=18,c='black',zorder=5)
    ax.set_aspect('equal',adjustable='datalim')

def median_path(ids):
    # Relative-duration interpolation preserves geometry but is not a fixed-time path.
    paths=[]
    for i in ids:
        a=TRACKS[i]
        paths.append(np.column_stack([np.interp(np.linspace(0,1,61),np.linspace(0,1,len(a)),a[:,j]) for j in [0,1]]))
    return np.median(paths,axis=0)

def overlay(ax, ids, color=COLORS[0], cap=250, label='Pointwise median'):
    chosen=np.random.default_rng(20261025).choice(ids,min(cap,len(ids)),replace=False)
    for i in chosen:
        a=TRACKS[i]; ax.plot(a[:,0],a[:,1],color=color,alpha=.09,lw=.65)
    m=median_path(ids); ax.plot(m[:,0],m[:,1],color=color,lw=2.6,label=label)
    return len(chosen)

def features(g):
    g=g.sort_values('frame_id'); x=g.x.to_numpy(); y=g.y.to_numpy(); frames=g.frame_id.to_numpy()
    right=g.play_direction.iloc[0]=='right'
    xx=x if right else 120-x; yy=y if right else 53.3-y
    side=1 if yy[0]>=53.3/2 else -1
    lateral=(yy-yy[0])*side; depth=xx-xx[0]
    t=(frames-frames[0])/10
    sx=gaussian_filter1d(depth,1); sy=gaussian_filter1d(lateral,1)
    # Compare headings over 0.3s before/after candidate turns; >1yd net movement each.
    candidates=[]
    for k in range(3,len(g)-3):
        u=np.array([sx[k]-sx[k-3],sy[k]-sy[k-3]])
        v=np.array([sx[k+3]-sx[k],sy[k+3]-sy[k]])
        if np.linalg.norm(u)>=.6 and np.linalg.norm(v)>=.6:
            angle=np.degrees(np.arctan2(u[0]*v[1]-u[1]*v[0],np.dot(u,v)))
            if abs(angle)>=35 and t[k]>=.3:
                candidates.append((abs(angle),k,angle))
    # Strongest observed turn is an exploratory proxy; not a charted route break.
    best=max(candidates) if candidates else None
    k=best[1] if best else None
    rawstep=np.hypot(np.diff(x),np.diff(y))
    d=dict(nfl_id=int(g.nfl_id.iloc[0]),player_name=g.player_name.iloc[0],position=g.player_position.iloc[0],direction=g.play_direction.iloc[0],
        frames=len(g),start_frame=int(frames[0]),end_frame=int(frames[-1]),observed_span_s=float(t[-1]),
        max_speed_yd_s=float(g.s.max()),max_depth_yd=float(depth.max()),end_depth_yd=float(depth[-1]),end_lateral_yd=float(lateral[-1]),
        initial_width_yd=float(abs(yy[0]-26.65)),
        alignment_proxy='outside (>=10 yd from center)' if abs(yy[0]-26.65)>=10 else 'interior (<10 yd from center)',
        break_depth_yd=float(depth[k]) if k is not None else np.nan,break_time_from_window_start_s=float(t[k]) if k is not None else np.nan,
        break_angle_deg=float(best[2]) if best else np.nan,turn_detected=best is not None,
        short_window=bool(t[-1]<1.5),late_turn=bool(k is not None and t[-1]-t[k]<=.5),
        returns_toward_qb=bool(depth.max()-depth[-1]>2),duplicate_frames=bool(g.frame_id.duplicated().any()),
        frame_gap=bool(np.any(np.diff(frames)!=1)),out_of_bounds=bool(np.any((x<0)|(x>120)|(y<0)|(y>53.4))),
        implausible_step=bool(np.any(rawstep>2)),median_forward_speed_yd_s=float(np.median(np.diff(depth)*10)))
    return d,np.column_stack([lateral,depth,t])

def main(args):
    global OUT, TRACKS
    OUT=HERE/'route_visualizations'; OUT.mkdir(exist_ok=True)
    TRACKS={}; raw=pd.read_csv(HERE/'supplementary_data.csv',low_memory=False)
    assert not raw.duplicated(['game_id','play_id']).any()
    meta=raw.set_index(['game_id','play_id']); rows=[]; examples={}; audit=[]
    cols=['game_id','play_id','nfl_id','frame_id','play_direction','absolute_yardline_number','player_name','player_position','player_side','player_role','x','y','s']
    for file in sorted(args.tracking_root.glob('input_*.csv')):
        data=pd.read_csv(file,usecols=cols)
        play_groups=data.groupby(['game_id','play_id'],sort=False)
        target=data[data.player_role.eq('Targeted Receiver')]
        target_counts=target.groupby(['game_id','play_id']).nfl_id.nunique()
        for (game,play),g in target.groupby(['game_id','play_id'],sort=False):
            if (game,play) not in meta.index: continue
            r=meta.loc[(game,play)]; label=r.route_of_targeted_receiver
            if label not in ROUTES: continue
            if target_counts.loc[(game,play)]!=1:
                audit.append({'game_id':game,'play_id':play,'issue':'multiple_targeted_receivers'}); continue
            f,path=features(g); i=len(rows)
            f.update(game_id=int(game),play_id=int(play),route=label,coverage=r.team_coverage_man_zone,
                formation_alignment=r.receiver_alignment,penalty_nullified=r.play_nullified_by_penalty)
            # Distance is computed in raw coordinates at the target's final observed input frame.
            play_data=play_groups.get_group((game,play))
            defenders=play_data[play_data.frame_id.eq(f['end_frame'])&play_data.player_side.eq('Defense')]
            end=g.sort_values('frame_id').iloc[-1]
            dist=np.hypot(defenders.x-end.x,defenders.y-end.y)
            f['separation_end_coverage_yd']=float(dist.min()) if len(dist) else np.nan
            db=defenders.player_position.isin(['CB','DB','FS','SS','S'])
            f['separation_end_db_yd']=float(dist[db].min()) if db.any() else np.nan
            rows.append(f); TRACKS[i]=path
            # Keep compact raw play paths for selecting animation examples after the scan.
        print(file.name, 'cumulative matched routes',len(rows),flush=True)
    df=pd.DataFrame(rows)
    assert len(df)>0 and not df.duplicated(['game_id','play_id']).any()
    df.to_csv(OUT/'route_features.csv',index=False)
    counts=raw.route_of_targeted_receiver.value_counts().reindex(ROUTES).rename('supplementary_count').to_frame()
    counts['matched_tracking_count']=df.route.value_counts().reindex(ROUTES).fillna(0).astype(int)
    counts['unmatched_count']=counts.supplementary_count-counts.matched_tracking_count
    counts.to_csv(OUT/'route_counts.csv',index_label='route')
    pd.DataFrame(audit,columns=['game_id','play_id','issue']).to_csv(OUT/'excluded_target_audit.csv',index=False)
    metrics=['break_depth_yd','break_time_from_window_start_s','break_angle_deg','max_speed_yd_s','observed_span_s','max_depth_yd','separation_end_coverage_yd']
    summary=[]
    for r in ROUTES:
        s=df[df.route.eq(r)]; row={'route':r,'source_n':int(counts.loc[r,'supplementary_count']),'tracking_n':len(s),'turn_detected_n':int(s.turn_detected.sum())}
        for c in metrics:
            row[c+'_median']=s[c].median(); row[c+'_q25']=s[c].quantile(.25); row[c+'_q75']=s[c].quantile(.75)
        row['verified_snap_to_throw_s']=np.nan
        summary.append(row)
    summary=pd.DataFrame(summary); summary.to_csv(OUT/'descriptive_statistics.csv',index=False)
    fig,axes=plt.subplots(3,4,figsize=(17,16),layout='constrained')
    for r,ax in zip(ROUTES,axes.flat):
        ids=df.index[df.route.eq(r)].to_numpy(); n=overlay(ax,ids)
        axis(ax,f'{r} | n={len(ids):,} | {n} shown'); ax.set(xlim=(-24,30),ylim=(-8,42))
        ax.set_aspect('equal',adjustable='box'); ax.legend(fontsize=7,loc='upper left')
        f,a=plt.subplots(figsize=(7,8),layout='constrained'); overlay(a,ids); axis(a,f'{r}: {n} observed paths / {len(ids):,} matched targets')
        a.legend(); f.suptitle('Thin: sampled routes | Thick: median at relative elapsed duration',fontsize=10)
        save(f,'route_'+r.lower())
    fig.suptitle('12 targeted-receiver route classes | 2023 weeks 1–18\nOrigin = first observed position; offense upfield; outward = positive lateral\n250 paths/class maximum; median uses all matched routes, resampled by relative duration',fontsize=15)
    save(fig,'all_12_routes')
    groups=[['SLANT','IN'],['HITCH','OUT'],['POST','CORNER','GO'],['FLAT','ANGLE','WHEEL']]
    fig,axes=plt.subplots(2,2,figsize=(13,14),layout='constrained')
    for ax,group in zip(axes.flat,groups):
        for r,c in zip(group,COLORS):
            overlay(ax,df.index[df.route.eq(r)].to_numpy(),c,70,r+' median')
        axis(ax,' vs. '.join(group)); ax.legend(fontsize=9)
    fig.suptitle('Confusable classes | 70 paths/class maximum; median uses all matched paths')
    save(fig,'confusable_routes')
    # Feature distributions and descriptive subtypes, without claiming latent clusters.
    fig,axes=plt.subplots(3,4,figsize=(17,11),layout='constrained'); bands=[]; modes=[]
    for ax,r in zip(axes.flat,ROUTES):
        s=df[df.route.eq(r)]; v=s.max_depth_yd
        ax.hist(v,bins=np.arange(-5,56,2),color=COLORS[0],alpha=.65)
        ax.set(title=f'{r} | observed maximum depth',xlabel='Depth from window start (yd)',ylabel='Targeted routes (count)')
        for lo,hi,name in [(-np.inf,5,'<5 yd'),(5,10,'5–<10 yd'),(10,15,'10–<15 yd'),(15,np.inf,'>=15 yd')]:
            n=int(((v>=lo)&(v<hi)).sum()); bands.append({'route':r,'depth_band':name,'n':n,'share':n/len(s)})
        grid=np.linspace(max(-5,v.min()),min(55,v.max()),300)
        density=gaussian_kde(v)(grid); peaks,_=find_peaks(density,prominence=density.max()*.12,distance=25)
        modes.append({'route':r,'exploratory_density_peaks_yd':'; '.join(f'{grid[p]:.1f}' for p in peaks),'peak_count':len(peaks)})
    save(fig,'depth_subtypes'); pd.DataFrame(bands).to_csv(OUT/'depth_subtype_bands.csv',index=False); pd.DataFrame(modes).to_csv(OUT/'depth_density_peaks.csv',index=False)
    for context,levels,title in [('position',['WR','TE','RB'],'Position'),('alignment_proxy',sorted(df.alignment_proxy.unique()),'Initial-width proxy (not verified slot/outside)'),('coverage',['MAN_COVERAGE','ZONE_COVERAGE'],'Charted man / zone coverage')]:
        fig,axes=plt.subplots(3,4,figsize=(17,15),layout='constrained')
        for ax,r in zip(axes.flat,ROUTES):
            for level,c in zip(levels,COLORS):
                ids=df.index[df.route.eq(r)&df[context].eq(level)].to_numpy()
                if len(ids)>=5:
                    m=median_path(ids); ax.plot(m[:,0],m[:,1],color=c,lw=2,label=f'{level} (n={len(ids)})')
            axis(ax,r)
            if ax.lines: ax.legend(fontsize=7)
        fig.suptitle(title+' | median trajectories, groups with at least 5 targets',fontsize=15)
        save(fig,'context_'+context)
    context_stats=df.groupby(['route','position','alignment_proxy','coverage'],dropna=False).agg(n=('route','size'),max_depth_median_yd=('max_depth_yd','median'),end_separation_median_yd=('separation_end_coverage_yd','median')).reset_index()
    context_stats.to_csv(OUT/'context_statistics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
    for ax,c,title in zip(axes,['separation_end_coverage_yd','separation_end_db_yd'],['Nearest supplied defender','Nearest supplied defensive back']):
        ax.boxplot([df.loc[df.route.eq(r),c].dropna() for r in ROUTES],tick_labels=ROUTES,showfliers=False,patch_artist=True,boxprops={'facecolor':'#b6cfdf'})
        ax.tick_params(axis='x',rotation=60); ax.set(title=title,xlabel='Targeted receiver route class',ylabel='Distance at final input frame (yd)')
    fig.suptitle('Separation at throw-time proxy | boxes = IQR, line = median; outlier points hidden')
    save(fig,'separation_boxplots')
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for ax,group in zip(axes.flat,groups):
        for r,c in zip(group,COLORS):
            s=df[df.route.eq(r)]; ax.scatter(s.max_depth_yd,s.end_lateral_yd,alpha=.18,s=9,color=c,label=r)
        ax.set(title=' vs. '.join(group),xlabel='Maximum observed depth (yd)',ylabel='Final lateral displacement (yd; + outward)'); ax.legend()
    save(fig,'feature_overlap')
    quality=df.groupby('route')[['short_window','turn_detected','late_turn','returns_toward_qb','duplicate_frames','frame_gap','out_of_bounds','implausible_step']].sum().reindex(ROUTES)
    quality['n']=df.route.value_counts(); quality.to_csv(OUT/'quality_flags.csv')
    df.loc[df.short_window|df.late_turn|(df.route.eq('GO')&df.returns_toward_qb)].to_csv(OUT/'routes_for_review.csv',index=False)
    # Orientation audit preserves physical inward/outward and compares directions.
    orient=df.groupby(['route','direction']).agg(n=('route','size'),forward_speed_median=('median_forward_speed_yd_s','median'),end_lateral_median=('end_lateral_yd','median'),depth_median=('max_depth_yd','median')).reset_index()
    orient.to_csv(OUT/'orientation_audit.csv',index=False)
    fig,axes=plt.subplots(3,4,figsize=(17,14),layout='constrained')
    for ax,r in zip(axes.flat,ROUTES):
        for direction,c in zip(['left','right'],COLORS):
            ids=df.index[df.route.eq(r)&df.direction.eq(direction)].to_numpy()
            if len(ids):
                m=median_path(ids); ax.plot(m[:,0],m[:,1],c=c,lw=2,label=f'{direction} (n={len(ids)})')
        axis(ax,r); ax.legend(fontsize=8)
    fig.suptitle('Orientation check | median normalized paths by original play direction')
    save(fig,'orientation_check')
    # Three high and three low separation examples, same route where possible.
    candidates=df[df.route.eq('GO')&df.observed_span_s.ge(2)&~df.frame_gap&~df.implausible_step].dropna(subset=['separation_end_coverage_yd'])
    chosen=pd.concat([candidates.nsmallest(3,'separation_end_coverage_yd').assign(example_group='low'),candidates.nlargest(3,'separation_end_coverage_yd').assign(example_group='high')])
    chosen.to_csv(OUT/'animation_examples.csv',index=False)
    animate_examples(chosen,args)
    write_report(df,summary,counts,quality,modes)
    print('DONE',OUT,flush=True)

def animate_examples(chosen,args):
    wanted={(int(s.game_id),int(s.play_id)):(i,s) for i,s in chosen.iterrows()}
    for file in sorted(args.tracking_root.glob('input_*.csv')):
        data=pd.read_csv(file,usecols=['game_id','play_id','nfl_id','frame_id','play_direction','player_side','player_role','x','y'])
        keys=pd.MultiIndex.from_frame(data[['game_id','play_id']]); data=data[keys.isin(wanted)]
        for key,g in data.groupby(['game_id','play_id']):
            i,s=wanted[key]; target=g[g.player_role.eq('Targeted Receiver')].sort_values('frame_id')
            right=target.play_direction.iloc[0]=='right'
            g=g.copy(); g['fx']=g.x if right else 120-g.x; g['fy']=g.y if right else 53.3-g.y
            fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
            ax.set(xlim=(max(0,g.fx.min()-3),min(120,g.fx.max()+3)),ylim=(0,53.3),xlabel='Downfield field coordinate (yd; offense right)',ylabel='Cross-field coordinate (yd)')
            ax.set_aspect('equal'); receiver=ax.scatter([],[],s=75,color=COLORS[0],label='Target receiver'); defense=ax.scatter([],[],s=40,color=COLORS[1],label='Supplied defenders')
            line,=ax.plot([],[],color=COLORS[0],lw=1.5); link,=ax.plot([],[],color='#555555',ls='--'); caption=ax.text(.02,.98,'',transform=ax.transAxes,va='top',fontsize=9); ax.legend(loc='lower right')
            frames=sorted(target.frame_id.unique()); start=frames[0]
            def update(frame):
                now=g[g.frame_id.eq(frame)]; r=now[now.player_role.eq('Targeted Receiver')]; d=now[now.player_side.eq('Defense')]
                receiver.set_offsets(r[['fx','fy']].to_numpy()); defense.set_offsets(d[['fx','fy']].to_numpy())
                trail=g[g.player_role.eq('Targeted Receiver')&g.frame_id.le(frame)].sort_values('frame_id'); line.set_data(trail.fx,trail.fy)
                sep=np.nan
                if len(r) and len(d):
                    dd=np.hypot(d.fx-r.fx.iloc[0],d.fy-r.fy.iloc[0]); near=d.loc[dd.idxmin()]; sep=dd.min(); link.set_data([r.fx.iloc[0],near.fx],[r.fy.iloc[0],near.fy])
                caption.set_text(f'{s.example_group.upper()} final separation | GO | {key[0]} / {key[1]}\nObserved elapsed {(frame-start)/10:.1f}s | nearest defender {sep:.2f} yd')
                return receiver,defense,line,link,caption
            animation=FuncAnimation(fig,update,frames=frames,interval=100,blit=False)
            animation.save(OUT/f'animation_{s.example_group}_{key[0]}_{key[1]}.gif',writer=PillowWriter(fps=10)); plt.close(fig)
            print('animation',s.example_group,key,flush=True)

def write_report(df,summary,counts,quality,modes):
    go=df[df.route.eq('GO')].separation_end_coverage_yd.median(); flat=df[df.route.eq('FLAT')].separation_end_coverage_yd.median()
    subtype=pd.read_csv(OUT/'depth_subtype_bands.csv')
    cross=subtype[(subtype.route=='CROSS')&(subtype.depth_band=='<5 yd')].share.iloc[0]
    med=df.groupby('route')[['max_depth_yd','end_lateral_yd']].median()
    pair_notes=f"SLANT vs IN: median observed maximum depth {med.loc['SLANT','max_depth_yd']:.1f} vs {med.loc['IN','max_depth_yd']:.1f} yd helps, while median lateral endpoints are similar. HITCH vs OUT: depths {med.loc['HITCH','max_depth_yd']:.1f} vs {med.loc['OUT','max_depth_yd']:.1f} yd overlap; outward displacement adds information. GO vs HITCH is not cleanly separated by observed depth alone ({med.loc['GO','max_depth_yd']:.1f} vs {med.loc['HITCH','max_depth_yd']:.1f} yd). POST/CORNER/GO require direction and curvature as well as depth; early-ending tracks can hide the distinction. FLAT/ANGLE/WHEEL have depth and lateral variation that suggests multi-feature classification. These are descriptive clues, not held-out classification results."
    summary_text=f'''Route EDA — 2023 observed targeted-receiver routes

Coverage: {len(df):,} matched targets from {int(counts.supplementary_count.sum()):,} labeled supplementary plays; all 18 weekly input files. Matched tracking counts reproduce the supplied list exactly; the source CSV includes additional plays. Unmatched labels remain in route_counts.csv. Labels are joined only to Targeted Receiver, never to teammates. Nullified plays remain included and identified in route_features.csv.

Shapes and separability: {pair_notes} A median can smooth away a real break. Full-window features require an earlier fixed cutoff for predictive modeling.

Subtypes: Depth bands are descriptive, not validated clusters. {cross:.1%} of CROSS targets remain below 5 yd of maximum displacement from their first observed position. The chosen density-peak settings detect no class with multiple prominent depth modes, so these data do not establish distinct quick/deep subtypes. KDE peaks depend on bandwidth and window length. A longer route cut short by the pass can occupy a shallow band.

Context: Position and charted man/zone medians are provided with sample sizes. receiver_alignment is a play-level formation (e.g. 3x1), not the receiver's slot designation. Initial width >=10 yd from field center is an explicitly labeled outside proxy; interior includes backfield players. Do not interpret it as charted slot alignment.

Quality: {int(df.short_window.sum()):,} windows are shorter than 1.5s; {int((~df.turn_detected).sum()):,} routes lack an observed turn meeting the heuristic; {int(df.late_turn.sum()):,} detected turns occur in the final 0.5s. {int(((df.route=='GO')&df.returns_toward_qb).sum()):,} GO tracks retreat >2 yd after their maximum depth: review candidates, not proven mislabels. Duplicate frames: {int(df.duplicate_frames.sum())}; frame gaps: {int(df.frame_gap.sum())}; out-of-field tracks: {int(df.out_of_bounds.sum())}; >2yd steps: {int(df.implausible_step.sum())}. Left/right normalized median overlays and orientation_audit.csv support direction checks; distribution differences alone do not prove normalization errors.

Separation: Final-input nearest supplied defender median = {go:.2f} yd for GO versus {flat:.2f} yd for FLAT. These are selected target-only descriptive comparisons, confounded by depth, timing, coverage, and target selection; not causal route effects or player rankings. Six GO animations hold route class constant and illustrate low/high final separation; extremes are examples, not typical plays. Defensive-back-only distances are shown separately.

Timing limitation: No snap/pass event labels exist in these input files. Frame 1 is observed-window start; last input frame is an unverified throw-time proxy. Verified time to throw and snap-relative break time are unavailable. observed_span_s is NOT time to throw.
'''
    (OUT/'one_page_summary.txt').write_text(summary_text,encoding='utf-8')
    methods='''Normalization: raw official input only. Left-moving plays rotate x=120-x, y=53.3-y. Translate first receiver point to (0,0). Mirror lateral displacement by initial side of field so positive means toward that initial sideline. Longitudinal axis is downfield; no rescaling of yards. Median trajectories interpolate every path to 61 equally spaced fractions of its own observed duration, then take coordinatewise medians; these are not event-aligned or fixed-clock trajectories. Seed 20261025 samples at most 250 paths per class, while medians and statistics use all matched paths. Heuristic break: Gaussian smoothing sigma=1 frame; headings over 0.3s before/after a candidate, both vector lengths >=0.6 yd; choose largest absolute turn >=35 degrees and >=0.3s after window start. Signed angle positive=outward. No qualifying turn produces missing break features. This is an observed-turn proxy, not a validated route-break detector. Max speed uses supplied s in yd/s. Separation uses raw Euclidean distance to all supplied defenders at the target final input frame; missing defender sets yield missing values. DB subset: CB/DB/FS/SS/S. Depth bands use maximum observed displacement, not line-of-scrimmage depth. Density peaks are exploratory scipy gaussian_kde peaks (12% prominence; minimum grid separation 25).'''
    (OUT/'methodology.txt').write_text(methods,encoding='utf-8')
    display=pd.DataFrame({'Route':summary.route,'Source n':summary.source_n,'Tracking n':summary.tracking_n,'Turn n':summary.turn_detected_n})
    for key,label in [('break_depth_yd','Turn depth (yd)'),('break_time_from_window_start_s','Turn elapsed (s)'),('break_angle_deg','Turn angle (deg)'),('max_speed_yd_s','Max speed (yd/s)'),('observed_span_s','Observed span (s)'),('separation_end_coverage_yd','End separation (yd)')]:
        display[label]=[f'{m:.2f} [{q1:.2f}, {q3:.2f}]' if pd.notna(m) else 'Not detected' for m,q1,q3 in zip(summary[key+'_median'],summary[key+'_q25'],summary[key+'_q75'])]
    display['Snap-to-throw (s)']='Unavailable'
    def img(name):
        p=OUT/name; b=base64.b64encode(p.read_bytes()).decode(); mime='image/gif' if p.suffix=='.gif' else 'image/png'
        return f'<img loading="lazy" src="data:{mime};base64,{b}" alt="{html.escape(p.stem)}">'
    sections=[('<h2>12 actual route classes</h2><p>Thin paths: reproducible sample. Thick path: median of all matched routes. Each path ends where observation ends.</p>'+img('all_12_routes.png')),
        '<h2>Descriptive statistics</h2><p>Median [25th, 75th percentile]. Turn statistics use only detected turns. Observed span is not time to throw.</p><div class="table">'+display.to_html(index=False,escape=True)+'</div>']
    for name,title in [('confusable_routes','Direct route comparisons'),('feature_overlap','Two-feature overlap'),('depth_subtypes','Depth variation and possible subtypes'),('context_position','By position'),('context_alignment_proxy','By initial width proxy'),('context_coverage','By man / zone'),('separation_boxplots','Separation at the final input frame'),('orientation_check','Normalization check')]:
        sections.append(f'<h2>{title}</h2>'+img(name+'.png'))
    sections.append('<h2>Six separation animations</h2><p>Three low and three high final-separation GO examples, selected among windows >=2s. Blue = target; orange = supplied defenders; dotted line = nearest defender.</p><div class="animations">'+''.join(img(p.name) for p in sorted(OUT.glob('animation_*.gif')))+'</div>')
    body='<!doctype html><html><head><meta charset="utf-8"><title>12 route classes — observed tracking EDA</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#172b3b;max-width:1400px;margin:36px auto;padding:0 28px;background:#fafbfc}h1{font-size:30px}h2{margin-top:48px}img{width:100%;height:auto;background:white}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;font-size:12px;background:white}td,th{padding:9px;border-bottom:1px solid #dce2e6;text-align:left;white-space:nowrap}.table{overflow:auto}.animations{display:grid;grid-template-columns:1fr 1fr;gap:20px}details{margin:24px 0}p{max-width:1050px}</style></head><body><h1>12 route classes: observed paths and separation</h1><p>2023 season · official input tracking, weeks 1–18 · targeted receivers only</p><details open><summary>One-page findings and limitations</summary><pre>'+html.escape(summary_text)+'</pre></details>'+''.join(sections)+'<h2>Methods</h2><p>'+html.escape(methods)+'</p><p>Source: supplementary_data.csv in the parent directory; official input_2023_w01.csv through input_2023_w18.csv. No post-throw output tracking used. Reproduce with build_route_analysis.py; full statistics and review flags are adjacent CSV files.</p></body></html>'
    (OUT/'route_analysis.html').write_text(body,encoding='utf-8')
    manifest={'matched_routes':len(df),'source_labeled_routes':int(counts.supplementary_count.sum()),'class_counts':counts.reset_index(names='route').to_dict('records'),'tracking_source':str(ARGS.tracking_root),'supplementary_source':str(HERE/'supplementary_data.csv'),'seed':20261025,'plots_per_class':12,'animations':len(list(OUT.glob('animation_*.gif'))),'snap_to_throw_available':False}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-root',type=Path,required=True)
    ARGS=parser.parse_args(); main(ARGS)
