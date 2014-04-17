"""Interactive manual labelling of images"""
import matplotlib.pyplot as plt, numpy as np
import vidtrack
import my, my.plot
import scipy.misc

DEBUG = False
TRIM_STEP = 0.5

# Helper function for trimming c-limits
def trim_trans(int_trim):
    return 0.5 + 0.5 / (1 + np.exp(int_trim))


class Server:
    """Serves images, records labels. Stores data in a database.
    
    You provide the image filenames, etc. This iterates through them
    and reads/writes data from the database (a dict).
    
    Detection of clicks and rendering of objects is handled by a 
    child 'Graphics' object stored in self.gfx
    """
    def __init__(self, db_filename, image_filenames, image_full_filenames, 
        image_priorities=None, start_index=0, hide_title=False, trim=-2):
        """Initialize a new server object.
        
        This is the main object for interactively labeling images. Once
        it has been created, all of its methods are called upon
        graphics events detected by self.gfx
        
        image_filenames: List of short filenames
            These will serve as the keys in the database
            This could really be any key
        image_full_filenames: full path to location of each image
        image_priorities: integer priority of each image
            Highest priorities will be served first
        db: pickled dict of image_key -> image_info
            Each image_info is a dict {'left': (Lx,Ly), 'right': (Rx,Ry)}
        start_index:
            which image to serve first
        """
        # Store info about images
        self.image_filenames = np.asarray(image_filenames)
        self.image_full_filenames = np.asarray(image_full_filenames)
        if image_priorities is not None:
            self.image_priorities = np.asarray(image_priorities).astype(np.int)
        else:
            self.image_priorities = np.zeros(len(self.image_filenames)).astype(
                np.int)
        
        # This keeps track of which image we're handling
        self.image_idx = start_index
        
        # Load db
        self.db_filename = db_filename
        self.db = my.misc.pickle_load(db_filename)

        # Flag whether to display image name
        self.hide_title = hide_title

        # Create a graphics object to handle the graphics stuff
        self.gfx = Graphics(receiver=self, trim=trim)
        self.update_image()
    

    # Dispatch functions called whenever self.gfx detects a left or right click
    def on_left(self, click_data):
        """Save left position"""
        self.store_location(click_data['xd'], click_data['yd'], which='left')
        self.gfx.draw_circle(click_data['xd'], click_data['yd'], 'left')
    
    def on_right(self, click_data):
        """Save right position"""
        self.store_location(click_data['xd'], click_data['yd'], which='right')
        self.gfx.draw_circle(click_data['xd'], click_data['yd'], 'right')


    # Called when self.gfx detects middle click
    def on_middle(self, click_data):
        """New random image"""
        self.image_idx = self.choose_new_image('random new')
        self.update_image()


    # Called when self.gfx detects clicks outside the limits
    def on_left_outside_left(self, click_data):
        """Previous image"""
        self.image_idx = self.choose_new_image('previous')
        self.update_image()
    
    def on_left_outside_right(self, click_data):
        """Next image"""
        self.image_idx = self.choose_new_image('next')
        self.update_image()
    
    def on_left_outside_bottom(self, click_data):
        """Lower trim"""
        self.gfx.change_trim(-TRIM_STEP)

    def on_left_outside_top(self, click_data):
        """Raise trim"""
        self.gfx.change_trim(TRIM_STEP)

    def choose_new_image(self, which_image):
        """Chooses new image to serve up"""
        # Which image to go to
        if which_image == 'next':
            new_idx = np.mod(self.image_idx + 1, len(self.image_filenames))
        elif which_image == 'previous':
            new_idx = np.mod(self.image_idx - 1, len(self.image_filenames))
        elif which_image == 'random new':
            # Helper function to find images needing help
            def needs_work(fn):
                if fn not in self.db:
                    return True
                if 'left' not in self.db[fn] or 'right' not in self.db[fn]:
                    return True
                return False
            
            # Filter by those still needing work
            needs_work_mask = np.array([
                needs_work(fn) for fn in self.image_filenames], dtype=np.bool)
            
            # Identify the images to choose randomly from
            if self.image_priorities is None:
                # No priorities, choose from all
                choose_from = self.image_filenames[needs_work_mask]
            else:
                # Choose only those with high priority
                highest = np.max(self.image_priorities[needs_work_mask])
                high_priority_mask = self.image_priorities == highest
                choose_from = self.image_filenames[
                    needs_work_mask & high_priority_mask]
            
            # Choose randomly among them
            idx = np.random.randint(0, len(choose_from))
            new_imagename = choose_from[idx]
            new_idx = list(self.image_filenames).index(new_imagename)
        else:
            new_idx = int(which_image)
        
        return new_idx
    
    def update_image(self):
        """Load image and data. Tell self.gfx to draw it"""
        # Get filenames
        fn_long = self.image_full_filenames[self.image_idx]
        fn_short = self.image_filenames[self.image_idx]
        
        # Load current image
        arr = scipy.misc.imread(fn_long, flatten=True)
        
        # Display, optionally with title
        self.gfx.update_image(arr, title='' if self.hide_title else fn_short)
        
        # Read from database and draw circles
        if fn_short in self.db:
            rec = self.db[fn_short]
            for which in ['left', 'right']:
                if which in rec:
                    rrec = rec[which]
                    cir = self.gfx.draw_circle(rrec[0], rrec[1], which)
    
    def store_location(self, xd, yd, which='left'):
        """Store coordinate in database"""
        # Current filename / key to db
        fn_short = self.image_filenames[self.image_idx]
        
        # Insert record into database if not already present
        if fn_short not in self.db:
            self.db[fn_short] = {}

        # Insert info about this location
        self.db[fn_short][which] = (xd, yd)
        if DEBUG:
            print "saving: ", fn_short, which, xd, yd
        
        # Save to disk
        my.misc.pickle_dump(self.db, self.db_filename)


class Graphics:
    """Object to detect clicks, draw objects, and keep track of handles
    
    Created by Server and reports back to it.
    """
    def __init__(self, receiver, trim=-2):
        """Create a new graphics object. Send events to 'receiver'"""
        # Plot the main figure window that everything will use
        self.f, self.ax = plt.subplots(figsize=(16, 12))
        
        # Connect
        self.f.canvas.mpl_connect('button_press_event', self.on_button_press)
        
        # Keep track of objects
        self.handles_d = {}
        
        # Who to send data to on mouse click
        # It must respond to on_middle, on_left, etc
        self.receiver = receiver
        self.trim = trim
        
        plt.show()

    def on_button_press(self, event):
        """Function to dispatch button presses to receiver's methods"""
        # Extract event info
        click_data = {
            'button': event.button,
            'x': event.x, 'y': event.y,
            'xd': event.xdata, 'yd': event.ydata} # None if not within limits
        click_data['in_limits'] = (
            click_data['xd'] is not None and click_data['yd'] is not None)

        if DEBUG:
            print "button press: ", click_data
        
        # First determine if inside or outside data limits
        if click_data['in_limits']:
            # Inside data limits, dispatch based on which button
            if click_data['button'] == 1:
                self.receiver.on_left(click_data)
            elif click_data['button'] == 2:
                self.receiver.on_middle(click_data)
            elif click_data['button'] == 3:
                self.receiver.on_right(click_data)
        else:
            # Outside data limits, dispatch based on which button
            if click_data['button'] == 1:
                # Left click outside limits
                # Convert to coordinates within figure, as 0-1 on both axes
                trans = self.f.transFigure.inverted()
                x2, y2 = trans.transform([click_data['x'], click_data['y']])
                
                # Dispatch depending on whether we're on the right or the left
                if x2 > .9:
                    self.receiver.on_left_outside_right(click_data)
                elif x2 < .1:
                    self.receiver.on_left_outside_left(click_data)
                elif y2 < .1:
                    self.receiver.on_left_outside_bottom(click_data)
                elif y2 > .1:
                    self.receiver.on_left_outside_top(click_data)
                else:
                    pass
            else:
                # Some other type of click outside the limits
                pass

    def draw_circle(self, xd, yd, which):
        """Draw a red or blue circle"""
        # Destory old
        self.destroy_handle(which)
        
        # Decide color
        if which == 'left':
            color = 'b'
        else:
            color = 'r'
        
        # Draw it
        obj, = self.ax.plot(xd, yd, color=color, marker='o')
        plt.draw_if_interactive()

        # Keep a record
        self.handles_d[which] = obj
        return obj
    
    def destroy_all_handles(self):
        """Destory all current graphic objects"""
        to_remove = self.handles_d.keys()
        for name in to_remove:
            self.destroy_handle(name)
    
    def destroy_handle(self, name):
        """Destory one graphics object"""
        # See if we have a record of it
        if name in self.handles_d:
            # Remove the handle from handles_d
            shape = self.handles_d.pop(name)
        
            # See if it's in the axis
            if shape in self.ax.get_children():
                # Try to remove it
                try:
                    shape.remove()
                except ValueError:
                    print "change_image: some error removing", shape, name

    def update_image(self, arr, title=''):
        """Draw a new image"""
        self.destroy_all_handles()
        
        # Display
        im = my.plot.imshow(arr, cmap=plt.cm.gray, ax=self.ax)
        self.handles_d['image'] = im
        self.ax.set_title(title)

        # Zoom
        self.ax.set_xlim((160, 320))
        self.ax.set_ylim((120, 0))
        
        # Trim
        my.plot.harmonize_clim_in_subplots(fig=self.f, 
            trim=trim_trans(self.trim))

        plt.draw_if_interactive()

    def change_trim(self, delta):
        """Increment or decrement color limits"""
        self.trim = self.trim - delta
        my.plot.harmonize_clim_in_subplots(fig=self.f, 
            trim=trim_trans(self.trim))
        plt.draw_if_interactive()



