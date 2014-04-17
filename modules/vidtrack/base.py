import my, my.misc, numpy as np, os, glob, pandas

class FileSchema:
    """Object to encapsulate the file schema for a single vidtrack session
    
    Returns filenames upon request, checks for existing files, etc.
    
    Each session is associated with a directory called 'full_path'.
    Within this directory is a set of image files, named something like
    '0035.png' where the number should be the behavioral trial number.
    
    Other files:
        video_filename : something ending in 'mp4'
        database_filename : session_name.pdict containing a pickled dict
    """
    def __init__(self, full_path):
        self.full_path = os.path.abspath(full_path)
        self.name = os.path.split(self.full_path)[1]
        self.IMAGE_CONSTRUCTION_STR = r'%04d.png'
        self.IMAGE_REGEX_STR = r'(\d+).png'
        
        # Find the video filename
        video_candidates = glob.glob(os.path.join(self.full_path, '*.mp4'))
        if len(video_candidates) == 0:
            self.video_filename = None
        elif len(video_candidates) == 1:
            self.video_filename = video_candidates[0]
        else:
            print "warning: multiple video files in %s" % self.full_path
            self.video_filename = video_candidates[0]
        
        # Find the image files
        self.image_numbers = self._load_image_numbers()
        self.image_filenames = map(self.trial2image_filename,   
            self.image_numbers)
        self.image_full_filenames = map(
            lambda s: os.path.join(self.full_path, s), self.image_filenames)
    
    def trial2image_filename(self, trial_number):
        return self.IMAGE_CONSTRUCTION_STR % trial_number
    
    def image_filename2trial(self, image_filename):
        # Apply regex
        res = my.misc.apply_and_filter_by_regex(self.IMAGE_REGEX_STR, 
            [image_filename])
        if len(res) == 0 or len(res) > 1:
            raise ValueError("%s doesn't uniquely match %s" % (image_filename,
                self.IMAGE_REGEX_STR))
        
        # Intify
        res_i = int(res[0])
        return res_i
    
    def _load_image_numbers(self):
        all_files = os.listdir(self.full_path)
        image_numbers_s = my.misc.apply_and_filter_by_regex(
            self.IMAGE_REGEX_STR, all_files, sort=True)
        image_numbers = np.array(map(int, image_numbers_s))
        return image_numbers
    
    @property
    def n2v_sync_filename(self):
        return os.path.join(self.full_path, 'N2V_SYNC')
    
    @property
    def v2n_sync_filename(self):
        return os.path.join(self.full_path, 'V2N_SYNC')
    
    @property
    def db_filename(self):
        return os.path.join(self.full_path, self.name + '.pdict')


class Session:
    """Object to handle reading and writing from vidtrack sessions
    
    Generally this is initialized from an existing session.
    To create a session, see create_session
    """
    def __init__(self, full_path=None, file_schema=None):
        """Initialize from a file schema (preferred)"""
        if file_schema is None:
            self.file_schema = FileSchema(full_path=full_path)
        else:
            self.file_schema = file_schema

    def write_n2v_sync(self, n2v_sync):
        """Write sync info as plaintext"""
        np.savetxt(self.file_schema.n2v_sync_filename, n2v_sync)
    
    def write_v2n_sync(self, v2n_sync):
        """Write sync info as plaintext"""
        np.savetxt(self.file_schema.v2n_sync_filename, v2n_sync)
    
    def save_db(self, db):
        """Save a new copy of the database"""
        my.misc.pickle_dump(db, self.file_schema.db_filename)
    
    def backup_db(self, suffix=None):
        """Save a backup of the current db"""
        if suffix is None:
            import datetime
            suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        os.system('cp %s %s.%s' % (self.file_schema.db_filename, 
            self.file_schema.db_filename, suffix))
    
    @property
    def db(self):
        """Return current copy of the database"""
        try:
            return my.misc.pickle_load(self.file_schema.db_filename)
        except IOError:
            return None

    @property
    def n2v_sync(self):
        res = None
        try:
            res = np.loadtxt(self.file_schema.n2v_sync_filename,
                dtype=np.float)
        except IOError:
            pass
        return res
    
    @property
    def v2n_sync(self):
        res = None
        try:
            res = np.loadtxt(self.file_schema.v2n_sync_filename,
                dtype=np.float)
        except IOError:
            pass
        return res    

def create_session(full_path, video_filename=None, v2n_sync=None, 
    n2v_sync=None, db=None):
    """Factory to create new vidtrack session"""
    # Create directory
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    
    # Create schema and session
    fs = FileSchema(full_path)
    session = Session(file_schema=fs)
    
    # Link video
    if video_filename is not None:
        video_filename = os.path.abspath(video_filename)
        short_filename = os.path.split(video_filename)[1]
        new_video_filename = os.path.join(full_path, short_filename)
        os.system('ln -s %s %s' % (video_filename, new_video_filename))
    
    # Write syncs
    if v2n_sync is not None:
        session.write_v2n_sync(v2n_sync)
    if n2v_sync is not None:
        session.write_n2v_sync(n2v_sync)

    return session


def dump_frames_by_trial_and_event(video_filename, event_times, trial_numbers,
    syncing_poly=None, latency=0., suffix=None, image_dir='.', filenamer=None,
    trial_stride=1, trial_offset=0):
    """Given an event or times, dump frames around each event
 
    video_filename : video to analyze
    event_times : array of times of events of interest in seconds
        A syncing polynomial ('syncing_poly') may be applied
        and a latency ('latency') in seconds may be added.
        Moreover, this array can be subindexed with 'trial_stride' and
        'trial_offset'.
        The image frame of the results will be written to disk.
    trial_numbers : array of ints, same length as event_times
        Used to label the images. Should be behavioral, generally.
    filenamer : function accepting 'image_dir', trial number, and a string
        called 'suffix' to name the file. See a default in this function.
    """
    # Apply the poly and the latency
    event_times = np.polyval(syncing_poly, event_times) + latency
    
    # Set up the iteration
    assert len(event_times) == len(trial_numbers)
    iterobj = zip(
        event_times[trial_offset::trial_stride],
        trial_numbers[trial_offset::trial_stride])

    # How to name files
    def default_filenamer(tnum, suffix):
        if suffix is None:
            return os.path.join(image_dir, '%04d.png' % tnum)
        else:
            return os.path.join(image_dir, '%04d_%s.png' % (tnum, suffix))
    if filenamer is None:
        filenamer = default_filenamer
    
    # Iterate over trials
    for ttime, tnum in iterobj:
        # Calculate which frame to dump
        frametime = ttime + latency
        if frametime < 0:
            continue
        
        # How to name file
        output_filename = filenamer(tnum, suffix)
        
        # Do it
        my.misc.frame_dump(filename=video_filename, frametime=frametime,
            output_filename=output_filename, meth='ffmpeg fast', verbose=True)


def vts_db2head_pos_df(vts, in_degrees=True):
    """Convert vidtrack-style db to dataframe of head position"""
    # Convert to dataframe and use file_schema to get btrial
    locdf = pandas.DataFrame.from_records(vts.db.values(), index=vts.db.keys())
    locdf['btrial'] = [vts.file_schema.image_filename2trial(fn) for
        fn in locdf.index]
    locdf = locdf.set_index('btrial').sort()
    
    # Parse out Lx, Rx, etc separately
    locdf['Lx'] = locdf['left'].apply(lambda arr: arr[0])
    locdf['Rx'] = locdf['right'].apply(lambda arr: arr[0])
    locdf['Ly'] = locdf['left'].apply(lambda arr: arr[1])
    locdf['Ry'] = locdf['right'].apply(lambda arr: arr[1])
    locdf = locdf.drop(['left', 'right'], axis=1)    

    # Add some more cols
    locdf['Mx'] = locdf[['Lx', 'Rx']].mean(1)
    locdf['My'] = locdf[['Ly', 'Ry']].mean(1)
    dx = locdf['Rx'] - locdf['Lx']
    dy = locdf['Ry'] - locdf['Ly']
    
    # This needs to be done more carefully -- I think one axis is reversed
    locdf['angl'] = np.arctan(dx / dy)
    if in_degrees:
        locdf['angl'] = locdf['angl'] * 180 / np.pi
    locdf['dist'] = np.sqrt(dx ** 2 + dy ** 2)
    
    return locdf