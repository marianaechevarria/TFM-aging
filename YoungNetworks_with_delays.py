from brian2 import *
import sys
import numpy as np
import time
import socket
from scipy.io import savemat
import multiprocessing as mp



start_time = time.time()   # returns the number of seconds passed since epoch
defaultclock.dt = 0.1*ms


#n_sims = 1                # number of simulations to run

# get integer from standard input to label output files
#par = int(sys.argv[1])


# define function to decode angles from spike counts
def decode(firing_rate, N_e): 
    angles = np.arange(0,N_e)*2*np.pi/N_e
    #R = np.sum(np.dot(firing_rate,np.exp(1j*angles)))/N_e
    R = np.sum(np.dot(firing_rate,np.exp(1j*angles)))/np.sum(firing_rate)  # by sara  (dot is the scalar product)
    angle = np.angle(R)    # sara: np.angle returns the angle of the complex argument (in radiand by default, betweeen (-pi, pi])
    modulus = np.abs(R)    # by sara
    if angle < 0:
        angle += 2*np.pi 
    return angle, modulus  # by sara 

# define function to get bump center diffusion from full rastergram (i,t)
def readout(i, t, sim_time, N_e):
    w1      = 100*ms                  # sara: time step of the sliding window
    w2      = 250*ms                  # sara: window size
    n_wins  = int((sim_time-w2)/w1)

    decs = []
    mods = []
    for ti in range(int(n_wins)):
        fr  = np.zeros(N_e)
        idx = ((t>ti*w1-w2/2) & (t<ti*w1+w2/2))
        ii  = i[idx]
        for n in range(N_e):
            fr[n] = sum(ii == n)            # sara: number of spikes for each neuron in the time interval defined by idx
        dec, mod = decode(fr, N_e)          # by sara
        decs.append(dec)
        mods.append(mod)                    # by sara
    return decs, n_wins, mods, w1, w2       # by sara


# simulation
def run_sims(i_sims):  
    
    # choose whether to store the state variable values (make_network=True) or restore them (faster)
    make_network = True
    
    # simulation parameters
    nstims=8   # number of possible equidistant locations for the stimulus  # by sara  
    
    stim_on=2*second                                       #as in H&M       !!modified
    stim_off=3*second                                      #as in H&M       !!modified
    runtime=7*second                                       #as in H&M       !!modified
    
    stimE=0.24*mV  #2.4*mV  #stimulus amplitude            #as in H&M       !!modified
    epsE=0.17               #stimulus width                #as in H&M       !!modified from Brian code: epsE=60 --> 60/360 = 0.17, as in H&M
    stimI=0*mV                                             #as in H&M       !!modified
    epsI=0                                                 #as in H&M 
    
    N=20000  # total number of neurons                     #as in H&M
    K=500    # total number of inputs                      #as in H&M
    
    tauE=20*ms                                  #as in H&M 
    tauI=10*ms                                  #as in H&M
    taua=3*ms  # AMPA synapse decay             #as in H&M
    taun=50*ms # NMDA synapse decay             #as in H&M
    taug=4*ms  # GABA synapse decay             #as in H&M
    
    Vt=20*mV          # spike threshold         #as in H&M
    Vr=-3.33*mV       # reset value             #as in H&M
    refE=0*ms         # refractory period       #as in H&M
    refI=0*ms         # refractory period       #as in H&M
    
    # parameters for short-term plasticity
    U=0.03                                      #as in H&M
    taud=200*ms                                 #as in H&M
    tauf=450*ms                                 #as in H&M

    # connectivity strengths    
    gEEA=533.3*mV*ms                              #as in H&M
    gEEN=490.64*mV*ms                             #as in H&M !!modified 
    gEIA=67.2*mV*ms                               #as in H&M
    gEIN=7.4*mV*ms                                #as in H&M
    gIE=-138.6*mV*ms                              #as in H&M
    gII=-90.6*mV*ms                               #as in H&M
    sigmaEE=30  # E-to-E footprint in degrees     #as in H&M !!modified 
    sigmaEI=35  # E-to-E footprint in degrees     #as in H&M !!modified 
    sigmaIE=30  # E-to-E footprint in degrees     #as in H&M !!modified 
    sigmaII=30  # E-to-E footprint in degrees     #as in H&M !!modified 

    # these are intermediate calculations needed for the equations below
    NE=int(ceil(0.8*N))   # number of excitatory neurons
    NI=int(floor(0.2*N))  # number of inhibitory neurons

    KE=int(ceil(0.8*K))   # number of excitatory inputs
    KI=int(floor(0.2*K))  # number of inhibitory inputs

    sigEE=sigmaEE/360.0
    sigEI=sigmaEI/360.0
    sigIE=sigmaIE/360.0
    sigII=sigmaII/360.0 

    gEEA=gEEA/sqrt(KE)/taua   #not as in H&M   !!modified 
    gEEN=gEEN/sqrt(KE)/taun   #not as in H&M   !!modified 
    gEIA=gEIA/sqrt(KE)/taua   #not as in H&M 
    gEIN=gEIN/sqrt(KE)/taun   #not as in H&M 
    gIE=gIE/sqrt(KI)/taug     #not as in H&M 
    gII=gII/sqrt(KI)/taug     #not as in H&M 
    
    stimE=stimE*sqrt(KE)      #as in H&M
    stimI=stimI*sqrt(KE)      #as in H&M


    # equations for each neuron
    eqs = '''
    Irec = gea+gen+gi + Ix     : volt
    dV/dt = (Irec-V+Iext)/tau  : volt (unless refractory)
    dgea/dt = -gea/taua        : volt
    dgen/dt = -gen/taun        : volt
    dgi/dt = -gi/taug          : volt
    tau                        : second
    Ix                         : volt
    Iext                       : volt

    ''' 
    
    reset = '''
    V = Vr
    '''
   
    # generate the two populations
    networkE = NeuronGroup(NE,model=eqs,threshold='V > Vt',reset=reset, refractory=refE, method='exact', name='networkE')  # method added by Sara
    networkI = NeuronGroup(NI,model=eqs,threshold='V > Vt',reset='V = Vr', refractory=refI, method='exact', name='networkI')  # method added by Sara
    
    # create synapses 
    C1 = Synapses(networkE, networkE,
                  model=''' w    : volt
                            dx/dt=(1-x)/taud : 1 (event-driven)
                            du/dt=(U-u)/tauf : 1 (event-driven) ''',
                  on_pre='''gea += w*u*x
                            x *= 1-u
                            u += U*(1-u) ''', name='C1') 
    C2 = Synapses(networkE, networkE,
                  model=''' w    : volt
                            dx/dt=(1-x)/taud : 1 (event-driven)
                            du/dt=(U-u)/tauf : 1 (event-driven) ''',
                  on_pre='''gen += w*u*x
                            x *= 1-u
                            u += U*(1-u) ''', name='C2')                     
    C3 = Synapses(networkE, networkI, 
                  model=''' w    : volt ''',                      
                  on_pre='''gea += w ''', name='C3')    
    C4 = Synapses(networkE, networkI,
                  model=''' w    : volt ''',   
                  on_pre='''gen += w ''', name='C4')                           
    C5 = Synapses(networkI, networkE, 'w: volt', on_pre='gi += w',name='C5')
    C6 = Synapses(networkI, networkI, 'w: volt', on_pre='gi += w',name='C6')

    if make_network:      
        seed(379)
        
        # connect synapses
        fE = float(KE)/float(NE)/np.sqrt(2*np.pi)
        fI = float(KI)/float(NI)/np.sqrt(2*np.pi)
        
        C1.connect(p = 'fE/sigEE  * exp(-(i/NE-j/NE)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) <= 0.5)) + fE/sigEE * exp(-(abs(i/NE-j/NE)-1)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) > 0.5)) ')
        C1.w = gEEA
        C1.u = U
        C1.x = 1
        #C1.delay = 'rand()*60*ms'
            
        C2.connect(p = 'fE/sigEE  * exp(-(i/NE-j/NE)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) <= 0.5)) + fE/sigEE * exp(-(abs(i/NE-j/NE)-1)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) > 0.5)) ')
        C2.w = gEEN
        C2.u = U
        C2.x = 1
        #C2.delay = 'rand()*60*ms'
            
        C3.connect(p = 'fE/sigEI * exp(-(i/NE-j/NI)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) <= 0.5)) + fE/sigEI * exp(-(abs(i/NE-j/NI)-1)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) > 0.5)) ')
        C3.w = gEIA
        #C3.delay = 'rand()*60*ms'
        
        C4.connect(p = 'fE/sigEI * exp(-(i/NE-j/NI)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) <= 0.5)) + fE/sigEI * exp(-(abs(i/NE-j/NI)-1)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) > 0.5)) ')
        C4.w = gEIN
        #C4.delay = 'rand()*60*ms'
        
        C5.connect(p = 'fI/sigIE * exp(-(i/NI-j/NE)**2 / (2*sigIE**2)) * (int( (i/NI-j/NE)*sign(i/NI-j/NE) <= 0.5)) + fI/sigIE * exp(-(abs(i/NI-j/NE)-1)**2 / (2*sigIE**2)) * (int( (i/NI-j/NE)*sign(i/NI-j/NE) > 0.5)) ')
        C5.w = gIE
            
        C6.connect(p = 'fI/sigII * exp(-(i/NI-j/NI)**2 / (2*sigII**2)) * (int( (i/NI-j/NI)*sign(i/NI-j/NI) <= 0.5)) + fI/sigII * exp(-(abs(i/NI-j/NI)-1)**2 / (2*sigII**2)) * (int( (i/NI-j/NI)*sign(i/NI-j/NI) > 0.5)) ')
        C6.w = gII

        seed()
        #store(filename = 'my_network')
        
    else:      
        restore(filename = 'my_network',restore_random_state=False)
        C1.connect(False)
        C2.connect(False)
        C3.connect(False)
        C4.connect(False)
        C5.connect(False)
        C6.connect(False)
    
        
    # initialize parameters for the two populations
    networkE.tau = tauE                       #as in H&M
    networkI.tau = tauI                       #as in H&M
    networkE.Ix = 1.66*sqrt(KE)*mV            #as in H&M   
    networkI.Ix = 1.85*0.83*sqrt(KE)*mV       #as in H&M  !!modified   
    networkE.V = Vr + rand(NE)*(Vt - Vr)      # random initial conditions for V in each trial (E neurons)
    networkI.V = Vr + rand(NI)*(Vt - Vr)      # random initial conditions for V in each trial (I neurons)
    networkE.Iext = 0*mV                      #as in H&M
    networkI.Iext = 0*mV                      #as in H&M

    # monitor voltage, currents and spikes
    ME = StateMonitor(networkE, ('V','gea','gen','gi','Ix','Iext'), record=np.arange(0,NE,800))
    spikesE = SpikeMonitor(networkE)
    spE_i = spikesE.i
    spE_t = spikesE.t 
    spE_count = spikesE.count  
    MI = StateMonitor(networkI, ('V','gea','gen','gi','Ix'), record=np.arange(0,NI,400))
    spikesI = SpikeMonitor(networkI)
    spI_i = spikesI.i
    spI_t = spikesI.t
    spI_count = spikesI.count
       
    # run the simulation during the pre-cue period
    run(stim_on,report='text')


    ### CUE PERIOD
    # define the stimulus location
    #Stim_loc = 4    #stimulus at 180º (0 <= stim_loc <= 7)     # by sara
    #stimat = Stim_loc/nstims * NE                              # by sara   
    stimat = randint(0,nstims)/nstims * NE   #stimulus location random within nstims possible equidistant locations
 
    # define the stimulus input
    posE = arange(NE)  
    inpE = stimE * exp(-0.5 * (posE/float(NE)-0.5)**2 / (epsE**2))   #as in "wm_model_Hansel_2013.py"
    networkE.Iext = np.roll(inpE,int(stimat-NE/2))                   #as in "wm_model_Hansel_2013.py"
    networkI.Iext = stimI                                            #as in H&M
    
    # run the simulation during the cue period
    run(stim_off-stim_on,report='text')
    
     
    ### DELAY PERIOD
    # remove the stimulus input
    networkE.Iext = 0*mV
    networkI.Iext = 0*mV
    
    # monitor spikes during the delay
    # spikesE_d = SpikeMonitor(networkE)
    # spE_count_d = spikesE_d.count  
    # spikesI_d = SpikeMonitor(networkI)
    # spI_count_d = spikesI_d.count
    
    # run the simulation during the delay period
    run(runtime-stim_off,report='text')
 
  
    ## DO CALCULATIONS
    # get the decoded angle/modulus of the population vector through the whole simulation time
    popdec, nwins, popmod, window_t_step, window_size = readout(spE_i, spE_t, runtime, NE)  
    popdec = np.array(popdec)
    popmod = np.array(popmod)  
    
    # average firing rate for each E/I neuron during the delay period (option 1)
    # ratesE = spE_count_d/(runtime-stim_off)
    # ratesI = spI_count_d/(runtime-stim_off)
    
    # average firing rate for each E/I neuron during the delay period (option 2)
    fr_delay_E = np.zeros(NE)
    fr_delay_I = np.zeros(NI)
    idxE = ((spE_t > stim_off) & (spE_t <= runtime))
    idxI = ((spI_t > stim_off) & (spI_t <= runtime))
    spE_ii = spE_i[idxE]
    spI_ii = spI_i[idxI]
    for me in range(NE):
        fr_delay_E[me] = sum(spE_ii == me)         # sara: number of spikes for each E neuron during the delay period
    fr_delay_E = fr_delay_E / (runtime-stim_off)    
    for mi in range(NI):
        fr_delay_I[mi] = sum(spI_ii == mi)         # sara: number of spikes for each I neuron during the delay period
    fr_delay_I = fr_delay_I / (runtime-stim_off)

    
    ## SAVE DATA
    data1 = {'N_stim': nstims, 'stim': stimat, 'dec_angle': np.array(popdec), 'dec_modulus': np.array(popmod), 'sliding_window_size': window_size, 'window_time_step': window_t_step, 'number_windows': nwins}    
    savemat("decoding_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data1) 
    
    data2 = {'spikes_E_cells': np.array(spE_i), 'spike_times_E_cells': np.array(spE_t), 'spike_counts_E_cells': np.array(spE_count), 'spikes_I_cells': np.array(spI_i), 'spike_times_I_cells': np.array(spI_t), 'spike_counts_I_cells': np.array(spI_count)}    
    savemat("spikes_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data2)       
  
    data3 = {'fr_E' : np.array(fr_delay_E), 'fr_I' : np.array(fr_delay_I)}    
    savemat("rates_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data3)  
    
    data4 = {'V_E': ME.V, 'gea_E': ME.gea, 'gen_E': ME.gen, 'gi_E': ME.gi, 'Iext_E': ME.Iext, 'V_I': MI.V, 'gea_I': MI.gea, 'gen_I': MI.gen, 'gi_I': MI.gi}    
    savemat("system_variables_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data4) 
    


#####################################################################################################
#                                    RUN SIMULATIONS                                                #
#####################################################################################################           
num_trials = int(sys.argv[2])
myseeds = range(0, num_trials)                            
args = [(sd) for sd in myseeds]
numcores = mp.cpu_count()                                                    
pool = mp.Pool(processes=numcores, maxtasksperchild=1)                                                                          
pool.map(run_sims, args, chunksize=num_trials)  


#run_sims(n_sims)


print('all sims finished')
print(time.time() - start_time) 











































