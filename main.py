import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2
import pyttsx3
import math
import heapq
from queue import Queue
from modules.yolo_module import YOLODetector
from modules.midas_module import MiDaSDepth
from modules.smolvlm_module import describe_image, describe_video

class GridCell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.blocked = False
        self.parent = None
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')

    def __lt__(self, other):
        return self.f < other.f

class NavigationGrid:
    def __init__(self, rows, cols, clearance=1):
        self.rows = rows
        self.cols = cols
        self.grid = [[GridCell(c, r) for c in range(cols)] for r in range(rows)]
        self.start = (rows-1, cols//2)
        self.goal = None
        self.clearance = clearance
        self.debug_info = ""

    def reset_obstacles(self):
        for row in self.grid:
            for cell in row:
                cell.blocked = False

    def update_obstacles(self, depth_map, landmarks, threshold=3.0):
        self.reset_obstacles()
        
        if isinstance(depth_map, tuple):
            depth_map = depth_map[0]
        
        if depth_map is None or depth_map.size == 0:
            self.debug_info = "Invalid depth map received"
            return
            
        h, w = depth_map.shape
        obstacle_cells = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = int(c * w / self.cols)
                x2 = int((c+1) * w / self.cols)
                y1 = int(r * h / self.rows)
                y2 = int((r+1) * h / self.rows)
                
                cell_depth = np.median(depth_map[y1:y2, x1:x2])
                landmark_in_cell = False
                
                for lm in landmarks:
                    if 'cx' in lm and 'cy' in lm:
                        lm_x, lm_y = lm['cx'], lm['cy']
                    elif 'bbox' in lm:
                        bbox = lm['bbox']
                        lm_x = (bbox[0] + bbox[2]) // 2
                        lm_y = (bbox[1] + bbox[3]) // 2
                    else:
                        continue
                    
                    grid_x = int(lm_x * self.cols / w)
                    grid_y = int(lm_y * self.rows / h)
                    
                    if grid_x == c and grid_y == r:
                        landmark_in_cell = True
                        break
                
                if (cell_depth < threshold) or landmark_in_cell:
                    obstacle_cells.append((r, c))

        for r, c in obstacle_cells:
            for dr in range(-self.clearance, self.clearance+1):
                for dc in range(-self.clearance, self.clearance+1):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self.grid[nr][nc].blocked = True
        
        if self.goal:
            goal_r, goal_c = self.goal
            if 0 <= goal_r < self.rows and 0 <= goal_c < self.cols:
                self.grid[goal_r][goal_c].blocked = False
                
        start_r, start_c = self.start
        if 0 <= start_r < self.rows and 0 <= start_c < self.cols:
            self.grid[start_r][start_c].blocked = False
            
        self.debug_info = f"Detected {len(obstacle_cells)} obstacle cells"

    def find_path_a_star(self):
        if not self.goal:
            return []

        for row in self.grid:
            for cell in row:
                cell.parent = None
                cell.g = float('inf')
                cell.f = float('inf')
                cell.h = 0

        open_list = []
        start_cell = self.grid[self.start[0]][self.start[1]]
        start_cell.g = 0
        start_cell.h = self.heuristic(start_cell)
        start_cell.f = start_cell.h
        heapq.heappush(open_list, start_cell)

        closed_set = set()
        visited_count = 0

        while open_list:
            current = heapq.heappop(open_list)
            visited_count += 1
            
            if (current.y, current.x) == self.goal:
                return self.reconstruct_path(current)

            if (current.y, current.x) in closed_set:
                continue
                
            closed_set.add((current.y, current.x))

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
                x = current.x + dx
                y = current.y + dy
                
                if not (0 <= x < self.cols and 0 <= y < self.rows):
                    continue
                    
                neighbor = self.grid[y][x]
                
                if neighbor.blocked or (y, x) in closed_set:
                    continue

                move_cost = 1.414 if abs(dx)+abs(dy) == 2 else 1.0
                tentative_g = current.g + move_cost

                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    heapq.heappush(open_list, neighbor)
        
        return []

    def heuristic(self, cell):
        if not self.goal:
            return 0
            
        dx = abs(cell.x - self.goal[1])
        dy = abs(cell.y - self.goal[0])
        return 1.0 * max(dx, dy) + (1.414 - 1.0) * min(dx, dy)

    def reconstruct_path(self, current):
        path = []
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]

class VisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Vision Assistant")
        self.geometry("1400x800")
        self.update_interval = 100
        
        # Initialize modules
        self.yolo = YOLODetector()
        self.midas = MiDaSDepth()
        self.tts_lock = threading.Lock()

        
        # TTS management
        self.speech_queue = Queue()
        self.is_speaking = False
        self.tts_lock = threading.Lock()
        
        # Navigation state
        self.nav_landmarks = []
        self.nav_current_path = []
        self.nav_camera_active = False
        self.nav_cap = None
        self.nav_grid = NavigationGrid(20, 20, clearance=0)
        self.depth_threshold = 5.0
        self.last_depth_map = None
        
        # Description state
        self.desc_media_path = None
        self.desc_camera_active = False
        self.desc_cap = None
        
        # UI setup
        self.setup_ui()
        self.canvas_images = []

    def setup_ui(self):
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Navigation Section
        nav_frame = ttk.Frame(main_pane)
        main_pane.add(nav_frame)
        
        # Navigation Controls
        nav_control_frame = ttk.Frame(nav_frame)
        nav_control_frame.pack(pady=5, fill=tk.X)
        
        ttk.Button(nav_control_frame, text="Load Image", 
                 command=self.load_nav_image).pack(side=tk.LEFT, padx=5)
        self.nav_cam_btn = ttk.Button(nav_control_frame, text="Start Camera", 
                                    command=self.toggle_nav_camera)
        self.nav_cam_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_control_frame, text="Clear", 
                 command=self.clear_navigation).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(nav_control_frame, text="Depth Threshold:").pack(side=tk.LEFT, padx=5)
        self.depth_slider = ttk.Scale(nav_control_frame, from_=1.0, to=10.0, 
                                    orient=tk.HORIZONTAL, command=self.update_threshold)
        self.depth_slider.set(self.depth_threshold)
        self.depth_slider.pack(side=tk.LEFT, padx=5)
        
        self.obj_dropdown = ttk.Combobox(nav_control_frame, state="readonly")
        self.obj_dropdown.pack(side=tk.LEFT, padx=5)
        self.obj_dropdown.bind("<<ComboboxSelected>>", self.on_object_select)
        
        # Navigation Display
        nav_display_frame = ttk.Frame(nav_frame)
        nav_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image and Depth
        img_frame = ttk.Frame(nav_display_frame)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.nav_canvas = tk.Canvas(img_frame, width=640, height=360, bg="black")
        self.nav_canvas.pack(pady=5)
        self.depth_canvas = tk.Canvas(img_frame, width=320, height=180, bg="black")
        self.depth_canvas.pack(pady=5)
        
        # Map
        self.map_canvas = tk.Canvas(nav_display_frame, width=400, height=400, bg="white")
        self.map_canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Navigation Text
        self.nav_text = tk.Text(nav_frame, wrap="word", height=10)
        self.nav_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Description Section
        desc_frame = ttk.Frame(main_pane)
        main_pane.add(desc_frame)
        
        # Description Controls
        desc_control_frame = ttk.Frame(desc_frame)
        desc_control_frame.pack(pady=5, fill=tk.X)
        
        ttk.Button(desc_control_frame, text="Load Media", 
                 command=self.load_desc_media).pack(side=tk.LEFT, padx=5)
        self.desc_cam_btn = ttk.Button(desc_control_frame, text="Capture Media", 
                                     command=self.capture_desc_media)
        self.desc_cam_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(desc_control_frame, text="Clear", 
                 command=self.clear_description).pack(side=tk.LEFT, padx=5)
        
        # Description Display
        self.desc_canvas = tk.Canvas(desc_frame, width=640, height=360, bg="black")
        self.desc_canvas.pack(pady=10)
        
        # Description Text
        self.desc_text = tk.Text(desc_frame, wrap="word", height=15)
        self.desc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.draw_grid()

    def safe_speak(self, text):
        """Add text to speech queue safely"""
        with self.tts_lock:
            self.speech_queue.put(text)
            if not self.is_speaking:
                self.is_speaking = True
                tts_thread = threading.Thread(target=self.process_speech_queue)
                tts_thread.daemon = True
                tts_thread.start()

                
    def process_speech_queue(self):
        """Process speech queue in a dedicated thread"""
        while True:
            with self.tts_lock:
                if self.speech_queue.empty():
                    self.is_speaking = False
                    return
                
                text = self.speech_queue.get()

            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                self.after(0, self.show_tts_error, str(e))

    def show_tts_error(self, error):
        """Show error in UI thread"""
        self.nav_text.insert(tk.END, f"TTS error: {error}\n")


    def run_tts_loop(self):
        """Run the TTS event loop in a separate thread"""
        try:
            self.tts_engine.iterate()
            self.after(100, self.run_tts_loop)
        except RuntimeError:
            pass  # Loop already stopped

    def on_speech_end(self, name, completed):
        self.after(10, self.cleanup_tts)
        self.is_speaking = False
        self.process_speech_queue()

    def cleanup_tts(self):
        try:
            if self.tts_engine:
                self.tts_engine.endLoop()
                self.tts_engine.stop()
                del self.tts_engine
                self.tts_engine = None
        except Exception as e:
            pass

    def update_threshold(self, value):
        self.depth_threshold = float(value)
        if self.last_depth_map is not None:
            self.nav_grid.update_obstacles(self.last_depth_map, self.nav_landmarks, self.depth_threshold)
            self.recalculate_path()

    def recalculate_path(self):
        if self.nav_grid.goal:
            self.nav_current_path = self.nav_grid.find_path_a_star()
            self.update_map()
            self.nav_text.delete(1.0, tk.END)
            if self.nav_current_path:
                self.generate_navigation_instructions()
            else:
                self.nav_text.insert(tk.END, f"No path found!\n{self.nav_grid.debug_info}")

    def draw_grid(self):
        cell_size = 20
        self.map_canvas.delete("all")
        for r in range(self.nav_grid.rows):
            for c in range(self.nav_grid.cols):
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                self.map_canvas.create_rectangle(x1, y1, x2, y2, outline="gray")

    def update_map(self):
        cell_size = 20
        self.map_canvas.delete("all")
        
        for r in range(self.nav_grid.rows):
            for c in range(self.nav_grid.cols):
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                color = "red" if self.nav_grid.grid[r][c].blocked else "white"
                self.map_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
        
        if self.nav_current_path:
            for i in range(len(self.nav_current_path) - 1):
                x1, y1 = self.nav_current_path[i]
                x2, y2 = self.nav_current_path[i + 1]
                px1 = x1 * cell_size + cell_size//2
                py1 = y1 * cell_size + cell_size//2
                px2 = x2 * cell_size + cell_size//2
                py2 = y2 * cell_size + cell_size//2
                self.map_canvas.create_line(px1, py1, px2, py2, fill="blue", width=2)
                self.map_canvas.create_oval(px1-3, py1-3, px1+3, py1+3, fill="blue")
        
        start_r, start_c = self.nav_grid.start
        sx = start_c * cell_size + cell_size//2
        sy = start_r * cell_size + cell_size//2
        self.map_canvas.create_oval(sx-6, sy-6, sx+6, sy+6, fill="green")
        
        if self.nav_grid.goal:
            goal_r, goal_c = self.nav_grid.goal
            gx = goal_c * cell_size + cell_size//2
            gy = goal_r * cell_size + cell_size//2
            self.map_canvas.create_oval(gx-6, gy-6, gx+6, gy+6, fill="yellow")

    def load_nav_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ])
        if path:
            self.process_nav_image(path)

    def process_nav_image(self, path):
        self.clear_navigation()
        try:
            img = Image.open(path).convert("RGB").resize((640, 360))
            self.process_nav_frame(img, is_live=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def toggle_nav_camera(self):
        if self.nav_camera_active:
            self.stop_nav_camera()
        else:
            self.start_nav_camera()

    def start_nav_camera(self):
        self.nav_cap = cv2.VideoCapture(0)
        if not self.nav_cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        self.nav_camera_active = True
        self.nav_cam_btn.config(text="Stop Camera")
        self.update_nav_camera()

    def stop_nav_camera(self):
        self.nav_camera_active = False
        if self.nav_cap:
            self.nav_cap.release()
        self.nav_cam_btn.config(text="Start Camera")

    def update_nav_camera(self):
        if self.nav_camera_active:
            ret, frame = self.nav_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                self.process_nav_frame(pil_img, is_live=True)
            self.after(self.update_interval, self.update_nav_camera)

    def process_nav_frame(self, input_image, is_live=False):
        if isinstance(input_image, Image.Image):
            pil_img = input_image
            img_cv = np.array(pil_img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        else:
            pil_img = Image.fromarray(input_image)
            img_cv = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        results = self.yolo.detect(img_cv)
        self.nav_landmarks = results

        depth_map = self.midas.estimate_depth(pil_img)  
        self.last_depth_map = depth_map
        
        if isinstance(depth_map, tuple):
            depth_map_for_display = depth_map[0]
        else:
            depth_map_for_display = depth_map

        self.nav_grid.update_obstacles(depth_map, results, self.depth_threshold)
        
        if results and not self.nav_grid.goal:
            self.nav_grid.goal = self.find_goal_from_landmarks()
            if self.nav_grid.goal:
                self.nav_current_path = self.nav_grid.find_path_a_star()

        self.update_map()
        self.display_nav_image(pil_img, results)
        self.display_depth(depth_map_for_display)
        self.update_dropdown(results)

    def find_goal_from_landmarks(self):
        if self.nav_landmarks:
            landmark = self.nav_landmarks[0]
            if 'cx' in landmark and 'cy' in landmark:
                center_x, center_y = landmark['cx'], landmark['cy']
            elif 'bbox' in landmark:
                x1, y1, x2, y2 = landmark['bbox']
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
            else:
                return None
                
            return self.pixel_to_grid(center_x, center_y)
        return None

    def display_nav_image(self, input_image, detections):
        img_np = np.array(input_image)
        
        for obj in detections:
            if 'bbox' in obj:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = obj.get('label', 'Object')
                cv2.putText(img_np, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                cx, cy = obj.get('cx', (x1+x2)//2), obj.get('cy', (y1+y2)//2)
                cv2.circle(img_np, (cx, cy), 4, (0, 0, 255), -1)
        
        img = Image.fromarray(img_np)
        photo = ImageTk.PhotoImage(img)
        
        self.nav_canvas.delete("all")
        self.nav_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.nav_canvas.image = photo

    def display_depth(self, depth_map):
        if depth_map.min() != depth_map.max():
            normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        else:
            normalized = np.zeros_like(depth_map)
            
        depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        depth_img = Image.fromarray(depth_colored).resize((320, 180))
        depth_photo = ImageTk.PhotoImage(depth_img)
        
        self.depth_canvas.delete("all")
        self.depth_canvas.create_image(0, 0, image=depth_photo, anchor=tk.NW)
        self.depth_canvas.image = depth_photo

    def update_dropdown(self, results):
        dropdown_values = []
        for obj in results:
            label = obj.get('label', 'Unknown')
            dist = obj.get('dist', 'unknown')
            dropdown_values.append(f"{label} ({dist}m)")
            
        self.obj_dropdown['values'] = dropdown_values
        if results:
            self.obj_dropdown.current(0)

    def pixel_to_grid(self, x, y):
        grid_x = min(max(0, int(x / 640 * self.nav_grid.cols)), self.nav_grid.cols-1)
        grid_y = min(max(0, int(y / 360 * self.nav_grid.rows)), self.nav_grid.rows-1)
        return (grid_y, grid_x)

    def on_object_select(self, event):
        idx = self.obj_dropdown.current()
        if 0 <= idx < len(self.nav_landmarks):
            target = self.nav_landmarks[idx]
            
            if 'cx' in target and 'cy' in target:
                center_x, center_y = target['cx'], target['cy']
            elif 'bbox' in target:
                x1, y1, x2, y2 = target['bbox']
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
            else:
                self.nav_text.insert(tk.END, "Cannot determine target position!\n")
                return
                
            self.nav_grid.goal = self.pixel_to_grid(center_x, center_y)
            
            gy, gx = self.nav_grid.goal
            if self.nav_grid.grid[gy][gx].blocked:
                self.nav_grid.grid[gy][gx].blocked = False
                
            self.nav_current_path = self.nav_grid.find_path_a_star()
            self.update_map()
            
            self.nav_text.delete(1.0, tk.END)
            if self.nav_current_path:
                self.generate_navigation_instructions()
            else:
                self.nav_text.insert(tk.END, f"No path found!\n{self.nav_grid.debug_info}\n")

    def generate_navigation_instructions(self):
        if not self.nav_current_path:
            self.nav_text.insert(tk.END, "No path available.\n")
            return

        step_size = 0.5
        instructions = []
        current_dir = None
        current_distance = 0.0
        prev_point = self.nav_current_path[0]

        for i in range(1, len(self.nav_current_path)):
            x, y = self.nav_current_path[i]
            dx = x - prev_point[0]
            dy = y - prev_point[1]
            
            direction = self.get_direction_name(dx, dy)
            
            if direction == current_dir:
                current_distance += math.hypot(dx, dy) * step_size
            else:
                if current_dir is not None:
                    instructions.append(f"Move {current_distance:.1f} meters {current_dir}")
                current_dir = direction
                current_distance = math.hypot(dx, dy) * step_size
            
            prev_point = (x, y)

        if current_dir is not None:
            instructions.append(f"Move {current_distance:.1f} meters {current_dir}")
            instructions.append("Target reached")

        self.nav_text.delete(1.0, tk.END)
        self.nav_text.insert(tk.END, "\n".join(instructions))
        
        for step in instructions:
            self.safe_speak(step)

    def get_direction_name(self, dx, dy):
        if dx == 0 and dy > 0: return "south"
        if dx == 0 and dy < 0: return "north"
        if dx > 0 and dy == 0: return "east"
        if dx < 0 and dy == 0: return "west"
        if dx > 0 and dy > 0: return "southeast"
        if dx < 0 and dy > 0: return "southwest"
        if dx > 0 and dy < 0: return "northeast"
        if dx < 0 and dy < 0: return "northwest"
        return "forward"

    def load_desc_media(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Media files", "*.jpg *.jpeg *.png *.mp4 *.avi"),
            ("All files", "*.*")
        ])
        if path:
            self.process_desc_media(path)

    def process_desc_media(self, path):
        self.clear_description()
        try:
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(path).convert("RGB")
                img.thumbnail((640, 360))
                self.display_desc_image(img)
                description = describe_image(path)
                self.desc_text.insert(tk.END, f"Description:\n{description}\n")
                self.safe_speak(description)
            elif path.lower().endswith(('.mp4', '.avi')):
                # Extract and display first frame as thumbnail
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame)
                    pil_img.thumbnail((640, 360))
                    self.display_desc_image(pil_img)
                cap.release()
                
                # Get full video description
                description = describe_video(path)
                self.desc_text.insert(tk.END, f"Video Description:\n{description}\n")
                self.safe_speak(description)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process media: {str(e)}")

    def capture_desc_media(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            self.display_desc_image(pil_img)
            try:
                temp_path = "temp_capture.jpg"
                pil_img.save(temp_path)
                description = describe_image(temp_path)
                self.desc_text.insert(tk.END, f"Description:\n{description}\n")
                self.safe_speak(description)
            except Exception as e:
                messagebox.showerror("Error", f"Description failed: {str(e)}")

    def display_desc_image(self, img):
        img.thumbnail((640, 360))
        photo = ImageTk.PhotoImage(img)
        self.desc_canvas.delete("all")
        self.desc_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.desc_canvas.image = photo

    def clear_navigation(self):
        self.stop_nav_camera()
        self.nav_canvas.delete("all")
        self.depth_canvas.delete("all")
        self.nav_text.delete(1.0, tk.END)
        self.nav_landmarks = []
        self.nav_current_path = []
        self.nav_grid = NavigationGrid(20, 20, clearance=0)
        self.depth_slider.set(self.depth_threshold)
        self.update_map()

    def clear_description(self):
        self.desc_canvas.delete("all")
        self.desc_text.delete(1.0, tk.END)
        self.desc_media_path = None

    def clear(self):
        self.clear_navigation()
        self.clear_description()

if __name__ == "__main__":
    app = VisionApp()
    app.mainloop()

