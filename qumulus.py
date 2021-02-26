import numpy as np
import sys
from scipy.io.wavfile import write
import pennylane as qml

samplerate = 44100 #Frequecy in Hz

def get_wave(freq, duration=0.5):
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t) * (np.cos(2* np.pi * t / duration - np.pi) + 1)/2
    return wave

def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 261.63 #Frequency of Note C4
    note_freqs = {octave[i]: base_freq * pow(2,(i/12)) for i in range(len(octave))}        
    note_freqs[''] = 0.0 # silent note
    return note_freqs

def get_song_data(music_notes, tempo):
    note_freqs = get_piano_notes() # Function that we made earlier
    song = [get_wave(note_freqs[note], 60/tempo) for note in music_notes.split('-')]
    song = np.concatenate(song)
    return song

def numbers_to_notes(num_list, mapping):
    notes = ''
    for a in num_list[1:]:
        notes =  notes + '-' +str(mapping[a])
    return notes[1:]

def notes_to_numbers(notes, mapping):
    num_list = [mapping[note] for note in notes.split('-')]
    return num_list

#7 note
def train_and_generate_7(input_notes, reverse = False, num_layers = 4, output_name = 'output', output_len = 60, tempo = 120, output_notes = False, scale='C-D-E-F-G-A-B'):
    num_qubit = 1
    weights = np.random.uniform(low=-np.pi * 2, high=np.pi * 2, size=(num_layers, 7)) #assuming 2D data input
    batch_size = 1
    notes = 8 # scale + rest
    
    #angles and stuff for cubic setup
    a = 2*np.arcsin(1/np.sqrt(3))
    b = np.pi - a
    c = np.pi/3
    d = np.cos(a/2)**2
    e = np.cos(b/2)**2
    target_angles = [[0, 0], [a, 0], [a, 2*c], [a, 4*c], [b, c], [b, 3*c], [b, 5*c], [0, np.pi]] #note database, list of angle pairs, y rot followed by z rots
    fidelity_list_map = [[1, d, d, d, e, e, e, 0], 
                         [d, 1, e, e, d, 0, d, e],
                         [d, e, 1, e, d, d, 0, e],
                         [d, e, e, 1, 0, d, d, e],
                         [e, d, d, 0, 1, e, e, d],
                         [e, 0, d, d, e, 1, e, d],
                         [e, d, 0, d, e, e, 1, d],
                         [0, e, e, e, d, d, d, 1]]
    mapping = {7:''}
    mapping_2 = {'':7}
    l = scale.split('-')
    for a in range(len(l)):
        mapping[a] = l[a]
        mapping_2[l[a]] = a
    m_train = notes_to_numbers(input_notes, mapping_2)
    
    dev = qml.device("default.qubit", wires=num_qubit)
    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    def apply_data_layer(params, wires, data):
        qml.Rot(0, params[0, 0]*data[1] + params[0, 1], 
                params[1, 0]*data[0] + params[1, 1],  
                wires=wires)

    def apply_mixing_layer(params, wires):
        qml.Rot(params[0], 
                params[1],
                params[2],
                wires=wires)

    @qml.qnode(dev)
    def apply_layers(params, wires, data, target):
        #reshaping parameters for ease
        data_params = params[:,:4].reshape((len(params), 2, 2))
        mixing_params = params[:,4:].reshape((len(params), 3))
        for i in range(num_layers):
            angles = target_angles[data[i]]
            
            #for successive differences in angles
            #if i >0:
            #    angles = [angles[j] - target_angles[data[i-1]][j] for j in range(2)]
            
            apply_data_layer(data_params[i], wires=wires, data=angles)
            apply_mixing_layer(mixing_params[i], wires=wires)

        #for fidelity with target state:
        angles = target_angles[target] #fetching angles of the note
        qml.RZ(-angles[1], wires=wires)
        qml.RY(-angles[0], wires=wires)
        return qml.expval(qml.PauliZ(wires))

    def fidelity(x):
        return (1+x)/2

    #array of fidelities with different note states
    def fidelities(params, wires, data):
        fid = []
        for i in range(notes):
            x = apply_layers(params, wires=wires, data=data, target=i)
            fid.append(fidelity(x))
        return fid

    def fidelity_loss(labels, fid_list): #fidelity loss, unweighted but for multiple clases
        loss = 0.0
        for l, fids in zip(labels, fid_list):
            target_fid_list = fidelity_list_map[l]
            for a in range(notes):
                loss = loss + (target_fid_list[a] - fids[a])**2
        loss = loss / len(labels)
        loss = loss / notes
        return loss

    def cost(w, X, Y):    
        fidelity_lists = [fidelities(w, wires=0, data = x) for x in X]
        return fidelity_loss(Y, fidelity_lists)

    n = int(len(m_train[num_layers:]))
    weight = weights

    for a in range(n):
        seq = [m_train[a:a + num_layers]]
        if reverse:
            seq.reverse()
        weight = opt.step(lambda w: cost(w, seq, [m_train[a+num_layers]]), weight)

    print('Model trained successfully')
    
    sample = m_train[-num_layers:]
    l = output_len
    for a in range(l):
        f = fidelities(weight, wires=0, data=sample[-num_layers:])
        sample.append(f.index(max(f)))
    
    note_freqs = get_piano_notes()
    
    music_notes = numbers_to_notes(sample, mapping)
    data = get_song_data(music_notes, tempo)

    data = data * (26300/np.max(data)) # Adjusting the Amplitude (Optional)
    
    samplerate = 44100
    write('music/' + output_name + '.mp3', samplerate, data.astype(np.int16))
    
    if output_notes == True:
        with open('music/' + output_name + '.txt', 'w') as text_file:
            text_file.write(music_notes)

#12 note
def train_and_generate_12(input_notes, 
                          reverse = False,
                          num_layers = 4,
                          output_name = 'output',
                          output_len = 60,
                          tempo = 120,
                          output_notes = False):
    num_qubit = 1
    weights = np.random.uniform(low=-np.pi * 2, high=np.pi * 2, size=(num_layers, 7)) #assuming 2D data input
    batch_size = 1
    notes = 12 # scale + rest
    
    #angles and stuff for icosahedron setup
    phi = (1 + np.sqrt(5))/2
    x = np.arccos(1 - 2/(phi**2 + 1))
    a = np.pi * 36 / 180
    b = a * 2
    c = x
    f = np.pi - c
    d = np.cos(c/2)**2
    e = np.cos(f/2)**2
    target_angles = [[0, 0], [c, 0], [c, b], [c, 2*b],
                     [c, 3*b], [c, 4*b], [f, a], [f, a + b],
                     [f, a + 2*b], [f, a + 3*b], [f, a + 4*b], [np.pi, 0]] #note database, list of angle pairs, y rot followed by z rots
    
    fidelity_list_map = [[1, d, d, d, d, d, e, e, e, e, e, 0], 
                         [d, 1, d, e, e, d, d, d, e, 0, e, e],
                         [d, d, 1, d, e, e, e, d, d, e, 0, e],
                         [d, e, d, 1, d, e, 0, e, d, d, e, e],
                         [d, e, e, d, 1, d, e, 0, e, d, d, e],#5
                         [d, d, e, e, d, 1, d, e, 0, e, d, e],
                         [e, d, e, 0, e, d, 1, d, e, e, d, d],
                         [e, d, d, e, 0, e, d, 1, d, e, e, d],
                         [e, e, d, d, e, 0, e, d, 1, d, e, d],
                         [e, 0, e, d, d, e, e, e, d, 1, d, d],
                         [e, e, 0, e, d, d, d, e, e, d, 1, d],
                         [0, e, e, e, e, e, d, d, d, d, d, 1]]#12
    mapping = {}
    mapping_2 = {}
    l = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    for a in range(len(l)):
        mapping[a] = l[a]
        mapping_2[l[a]] = a
    m_train = notes_to_numbers(input_notes, mapping_2)
    
    dev = qml.device("default.qubit", wires=num_qubit)
    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    def apply_data_layer(params, wires, data):
        qml.Rot(0, params[0, 0]*data[1] + params[0, 1], 
                params[1, 0]*data[0] + params[1, 1],  
                wires=wires)

    def apply_mixing_layer(params, wires):
        qml.Rot(params[0], 
                params[1],
                params[2],
                wires=wires)

    @qml.qnode(dev)
    def apply_layers(params, wires, data, target):
        #reshaping parameters for ease
        data_params = params[:,:4].reshape((len(params), 2, 2))
        mixing_params = params[:,4:].reshape((len(params), 3))
        for i in range(num_layers):
            angles = target_angles[data[i]]
            
            #for successive differences in angles
            #if i >0:
            #    angles = [angles[j] - target_angles[data[i-1]][j] for j in range(2)]
            
            apply_data_layer(data_params[i], wires=wires, data=angles)
            apply_mixing_layer(mixing_params[i], wires=wires)

        #for fidelity with target state:
        angles = target_angles[target] #fetching angles of the note
        qml.RZ(-angles[1], wires=wires)
        qml.RY(-angles[0], wires=wires)
        return qml.expval(qml.PauliZ(wires))

    def fidelity(x):
        return (1+x)/2

    #array of fidelities with different note states
    def fidelities(params, wires, data):
        fid = []
        for i in range(notes):
            x = apply_layers(params, wires=wires, data=data, target=i)
            fid.append(fidelity(x))
        return fid

    def fidelity_loss(labels, fid_list): #fidelity loss, unweighted but for multiple clases
        loss = 0.0
        for l, fids in zip(labels, fid_list):
            target_fid_list = fidelity_list_map[l]
            for a in range(notes):
                loss = loss + (target_fid_list[a] - fids[a])**2
        loss = loss / len(labels)
        loss = loss / notes
        return loss

    def cost(w, X, Y):    
        fidelity_lists = [fidelities(w, wires=0, data = x) for x in X]
        return fidelity_loss(Y, fidelity_lists)

    n = int(len(m_train[num_layers:]))
    weight = weights

    for a in range(n):
        seq = [m_train[a:a + num_layers]]
        if reverse:
            seq.reverse()
        weight = opt.step(lambda w: cost(w, seq, [m_train[a+num_layers]]), weight)

    print('Model trained successfully')
    
    sample = m_train[-num_layers:]
    l = output_len
    for a in range(l):
        f = fidelities(weight, wires=0, data=sample[-num_layers:])
        sample.append(f.index(max(f)))
    
    note_freqs = get_piano_notes()
    
    music_notes = numbers_to_notes(sample, mapping)
    data = get_song_data(music_notes, tempo)

    data = data * (26300/np.max(data)) # Adjusting the Amplitude (Optional)
    
    samplerate = 44100
    write('music/' + output_name + '.mp3', samplerate, data.astype(np.int16))
    
    if output_notes == True:
        with open('music/' + output_name + '.txt', 'w') as text_file:
            text_file.write(music_notes)

# 5 note
def train_and_generate_5(input_notes, num_layers = 4,
                         reverse = False,
                          output_name = 'output',
                          output_len = 60,
                          tempo = 120,
                          output_notes = False,
                        scale = 'C-D-E-G-A'):
    num_qubit = 1
    weights = np.random.uniform(low=-np.pi * 2, high=np.pi * 2, size=(num_layers, 7)) #assuming 2D data input
    batch_size = 1
    notes = 6 # scale + rest
    
    #angles and stuff for icosahedron setup
    a = np.pi/2
    d = 1/2
    target_angles = [[0, 0], [a, 0], [a, a], [a, 2*a], [a, 3*a], [2*a, 0]] #note database, list of angle pairs, y rot followed by z rots
    
    fidelity_list_map = [[1, d, d, d, d, 0], 
                         [d, 1, d, 0, d, d],
                         [d, d, 1, d, 0, d],
                         [d, 0, d, 1, d, d],
                         [d, d, 0, d, 1, d],
                         [0, d, d, d, d, 1]]
    mapping = {5:''}
    mapping_2 = {'':5}
    l = scale.split('-')
    for a in range(len(l)):
        mapping[a] = l[a]
        mapping_2[l[a]] = a
    m_train = notes_to_numbers(input_notes, mapping_2)
    
    dev = qml.device("default.qubit", wires=num_qubit)
    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    def apply_data_layer(params, wires, data):
        qml.Rot(0, params[0, 0]*data[1] + params[0, 1], 
                params[1, 0]*data[0] + params[1, 1],  
                wires=wires)

    def apply_mixing_layer(params, wires):
        qml.Rot(params[0], 
                params[1],
                params[2],
                wires=wires)

    @qml.qnode(dev)
    def apply_layers(params, wires, data, target):
        #reshaping parameters for ease
        data_params = params[:,:4].reshape((len(params), 2, 2))
        mixing_params = params[:,4:].reshape((len(params), 3))
        for i in range(num_layers):
            angles = target_angles[data[i]]
            
            #for successive differences in angles
            #if i >0:
            #    angles = [angles[j] - target_angles[data[i-1]][j] for j in range(2)]
            
            apply_data_layer(data_params[i], wires=wires, data=angles)
            apply_mixing_layer(mixing_params[i], wires=wires)

        #for fidelity with target state:
        angles = target_angles[target] #fetching angles of the note
        qml.RZ(-angles[1], wires=wires)
        qml.RY(-angles[0], wires=wires)
        return qml.expval(qml.PauliZ(wires))

    def fidelity(x):
        return (1+x)/2

    #array of fidelities with different note states
    def fidelities(params, wires, data):
        fid = []
        for i in range(notes):
            x = apply_layers(params, wires=wires, data=data, target=i)
            fid.append(fidelity(x))
        return fid

    def fidelity_loss(labels, fid_list): #fidelity loss, unweighted but for multiple clases
        loss = 0.0
        for l, fids in zip(labels, fid_list):
            target_fid_list = fidelity_list_map[l]
            for a in range(notes):
                loss = loss + (target_fid_list[a] - fids[a])**2
        loss = loss / len(labels)
        loss = loss / notes
        return loss

    def cost(w, X, Y):    
        fidelity_lists = [fidelities(w, wires=0, data = x) for x in X]
        return fidelity_loss(Y, fidelity_lists)

    n = int(len(m_train[num_layers:]))
    weight = weights

    for a in range(n):
        seq = [m_train[a:a + num_layers]]
        if reverse:
            seq.reverse()
        weight = opt.step(lambda w: cost(w, seq, [m_train[a+num_layers]]), weight)

    print('Model trained successfully')
    
    sample = m_train[-num_layers:]
    l = output_len
    for a in range(l):
        f = fidelities(weight, wires=0, data=sample[-num_layers:])
        sample.append(f.index(max(f)))
    
    note_freqs = get_piano_notes()
    
    music_notes = numbers_to_notes(sample, mapping)
    data = get_song_data(music_notes, tempo)

    data = data * (26300/np.max(data)) # Adjusting the Amplitude (Optional)
    
    samplerate = 44100
    write('music/' + output_name + '.mp3', samplerate, data.astype(np.int16))
    
    if output_notes == True:
        with open('music/' + output_name + '.txt', 'w') as text_file:
            text_file.write(music_notes)