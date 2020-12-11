#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math
import glob
import os
import sys
import argparse
import logging
import random
from constants import *

try:
    sys.path.append(glob.glob('/home/alex/thesis_ssd/CARLA_nightly/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


from utils import vector3d_to_array, degrees_to_radians
from datadescriptor import KittiDescriptor
from dataexport import *
from bbox import create_kitti_datapoint
#from carla_utils import KeyboardHelper, MeasurementsDisplayHelper
from constants import *
import lidar_utils  # from lidar_utils import project_point_cloud
import time
from math import cos, sin, ceil
from image_converter import *

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 10)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        #self.frame=self.world.get_snapshot().frame
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

PHASE = "training"
OUTPUT_FOLDER = os.path.join("/home/alex/thesis_ssd/CARLA_nightly/test_test/_out1", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes', 'locational']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')
LOCATIONAL_PATH = os.path.join(OUTPUT_FOLDER, 'locational/{0:06}.txt')

def draw_image(surface, image, blend=False):
    #print(image.shape)
    array = np.frombuffer(image, dtype=np.dtype("uint8"))
    array = np.reshape(array, (WINDOW_HEIGHT, WINDOW_WIDTH, 3))
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False



def sensor_setting(world):
    """Make a CarlaSettings object with the settings we need.
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()

    """

    cam_rgb_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the cam_rgb_bp to set image resolution and field of view.
    cam_rgb_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
    cam_rgb_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
    cam_rgb_bp.set_attribute('fov', '90.0')
    # Set the time in seconds between sensor captures
    #print cam_rgb_bp.get_attribute('sensor_tick')
    #print "*)()()()()()()()()()()()"
    #cam_rgb_bp.set_attribute('sensor_tick', '1.0')
    # Provide the position of the sensor relative to the vehicle.
    rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=CAMERA_HEIGHT_POS),carla.Rotation(yaw=0,pitch=0))


    cam_depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
    # Modify the attributes of the cam_depth_bp to set image resolution and field of view.
    cam_depth_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
    cam_depth_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
    cam_depth_bp.set_attribute('fov', '90.0')
    # Set the time in seconCarlaSyncModeCarlaSyncModeds between sensor captures
    #cam_depth_bp.set_attribute('sensor_tick', '1.0')
    # Provide the position of the sensor relative to the vehicle.
    depth_transform = carla.Transform(carla.Location(x=0, y=0, z=CAMERA_HEIGHT_POS),carla.Rotation(yaw=0,pitch=0))


    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '40')
    lidar_bp.set_attribute('range', str(MAX_RENDER_DEPTH_IN_METERS*100))	#cm to m
    lidar_bp.set_attribute('points_per_second', '720000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '7')
    lidar_bp.set_attribute('lower_fov', '-16')
    #lidar_bp.set_attribute('sensor_tick', '0.0')
    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=LIDAR_HEIGHT_POS),carla.Rotation(yaw=0,pitch=0))

    # (Intrinsic) K Matrix
    # | f 0 Cu
    # | 0 f Cv
    # | 0 0 1
    # (Cu, Cv) is center of image
    k = np.identity(3)
    k[0, 2] = WINDOW_WIDTH_HALF
    k[1, 2] = WINDOW_HEIGHT_HALF
    f = WINDOW_WIDTH / \
        (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    
    camera_to_car_transform = get_matrix(rgb_transform)
    to_unreal_transform = get_matrix(carla.Transform(carla.Location(x=0, y=0, z=0),carla.Rotation(yaw=90,roll=-90)), -1.0,1.0,1.0)
    camera_to_car_transform = np.dot(camera_to_car_transform,to_unreal_transform) 
    
    lidar_to_car_transform = get_matrix(lidar_transform)
    to_unreal_transform2 = get_matrix(carla.Transform(carla.Location(x=0, y=0, z=0),carla.Rotation(yaw=90)), 1.0,1.0,-1.0)

    lidar_to_car_transform = np.dot(lidar_to_car_transform,to_unreal_transform2)

    #returning as matrices
    return k, cam_rgb_bp, cam_depth_bp, lidar_bp, camera_to_car_transform, lidar_to_car_transform, rgb_transform, depth_transform, lidar_transform


def get_matrix(transform,sc_x=1.0, sc_y=1.0,sc_z=1.0):
    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = sc_x*c_p * c_y
    matrix[0, 1] = sc_y*(c_y * s_p * s_r - s_y * c_r)
    matrix[0, 2] = -sc_z*(c_y * s_p * c_r + s_y * s_r)
    matrix[1, 0] = sc_x*s_y * c_p
    matrix[1, 1] = sc_y*(s_y * s_p * s_r + c_y * c_r)
    matrix[1, 2] = sc_z*(-s_y * s_p * c_r + c_y * s_r)
    matrix[2, 0] = sc_x*s_p
    matrix[2, 1] = -sc_y*(c_p * s_r)
    matrix[2, 2] = sc_z*(c_p * c_r)
    return matrix


def transform_points(points, txm_mat):
    """
    Given a 4x4 transformation matrix, transform an array of 3D points.
    Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
    """
    # Needed foramt: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.
    points = points.transpose()
    # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # Point transformation
    points = txm_mat * points
    # Return all but last row
    return points[0:3].transpose()

def generate_datapoints(world,image, intrinsic, extrinsic, depth_image,player, agents, gen_time):
    """ Returns a list of datapoints (labels cv2cv2cv2and such) that are generated this frame together with the main image image """
    datapoints = []
    image = image.copy()
    # Stores all datapoints for the current frames
    #print(agents, flush=True)
    if image is not None and gen_time:
        #for agent_id in agents:
        #    agent=world.get_actor(agent_id)
        for agent in world.get_actors():
            if "vehicle" not in agent.type_id:
                continue

            #if should_detect_class(agent) and GEN_DATA:
            if True:
                #print("asdsadsadsadsadsadsad", flush=True)
                image, kitti_datapoint = create_kitti_datapoint(
                    agent, intrinsic, extrinsic, image, depth_image, player)
                if kitti_datapoint:
                    datapoints.append(kitti_datapoint)
        if image is not None and datapoints is not None:
            return image, datapoints
        else:
            logging.debug(
                "Datapoints or Image is None during gen time")
    else:
        if image is not None:
            return image, datapoints
        else:
            logging.debug(
                "Datapoints or Image is None")



def processing(world,image_rgb, image_depth, image_lidar, intrinsic, player,agents,camera_to_car_transform,lidar_to_car_transform,gen_time):
     if image_rgb is not None and image_depth is not None:
        # Convert main image
        image = to_rgb_array(image_rgb)
        extrinsic=get_matrix(player.get_transform())*camera_to_car_transform

        # Retrieve and draw datapoints
        #IMAGE IS AN ARRAY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        image, datapoints = generate_datapoints(world,image, intrinsic, extrinsic, image_depth, player, agents,gen_time)
        
        #Lidar signal processing
        # Calculation to shift bboxes relative to pitch and roll of player
        rotation = player.get_transform().rotation
        pitch, roll, yaw = rotation.pitch, rotation.roll, rotation.yaw
        # Since measurements are in degrees, convert to radians

        pitch = degrees_to_radians(pitch)
        roll = degrees_to_radians(roll)
        yaw = degrees_to_radians(yaw)
        #print('pitch: ', pitch)
        #print('roll: ', roll)
        #print('yaw: ', yaw)

        # Rotation matrix for pitch
        rotP = np.array([[cos(pitch),            0,              sin(pitch)],
                         [0,            1,     0],
                         [-sin(pitch),            0,     cos(pitch)]])
        # Rotation matrix for roll
        rotR = np.array([[1,            0,              0],
                         [0,            cos(roll),     -sin(roll)],
                         [0,            sin(roll),     cos(roll)]])

        # combined rotation matrix, must be in order roll, pitch, yaw
        rotRP = np.matmul(rotR, rotP)
        # Take the points from the point cloud and transform to car space
        pc_arr_cpy=np.frombuffer(image_lidar.raw_data, dtype=np.dtype('f4'))
        #print(pc_arr.shape)
        pc_arr=np.reshape(pc_arr_cpy,(int(pc_arr_cpy.shape[0]/4),4))[:,:3].copy()
        #print(pc_arr.shape)
        pc_arr[:,[0,2]]=-pc_arr[:,[0,2]]
        pc_arr[:,[0,1]]=pc_arr[:,[1,0]]
        #print(pc_arr.shape)
        point_cloud = np.array(transform_points(
            pc_arr,lidar_to_car_transform))
        #print(lidar_to_car_transform)
        point_cloud[:, 2] -= LIDAR_HEIGHT_POS
        #print(point_cloud.shape)
        point_cloud = np.matmul(rotRP, point_cloud.T).T
        # print(self._lidar_to_car_transform.matrix)
        # print(self._camera_to_car_transform.matrix)

        # Draw lidar
        # Camera coordinate system is left, up, forwards
        if VISUALIZE_LIDAR:
            # Transform to camera space by the inverse of camera_to_car transform
            point_cloud_cam = transform_points(point_cloud,np.linalg.inv(camera_to_car_transform))
            point_cloud_cam[:, 1] += LIDAR_HEIGHT_POS
            image = lidar_utils.project_point_cloud(
                image, point_cloud_cam, intrinsic, 1)

        #determine whether to save data
        #TO_DO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if image is not None:
            return image, datapoints, point_cloud, extrinsic
        else:
            logging.debug(
                    "Image is None")


def save_training_files(player, image, datapoints, point_cloud,intrinsic,extrinsic,captured_frame_no):
    logging.warning("Test1 Attempting to save frame no: {}".format(
        captured_frame_no))
    groundplane_fname = GROUNDPLANE_PATH.format(captured_frame_no)
    lidar_fname = LIDAR_PATH.format(captured_frame_no)
    kitti_fname = LABEL_PATH.format(captured_frame_no)
    img_fname = IMAGE_PATH.format(captured_frame_no)
    calib_filename = CALIBRATION_PATH.format(captured_frame_no)
    loc_filename = LOCATIONAL_PATH.format(captured_frame_no)

    save_groundplanes(
        groundplane_fname, player, LIDAR_HEIGHT_POS)
    save_ref_files(OUTPUT_FOLDER, captured_frame_no)
    save_image_data(
        img_fname, to_rgb_array(image))
    save_kitti_data(kitti_fname, datapoints)
    save_lidar_data(lidar_fname, point_cloud,
                    LIDAR_HEIGHT_POS, LIDAR_DATA_FORMAT)
    save_calibration_matrices(
        calib_filename, intrinsic, extrinsic)
    save_locational(loc_filename,player)

def should_detect_class(agent):
    """ Returns true if the agent is of the classes that we want to detect.
        Note that Carla has class types in lowercase """
    return True in [class_type in agent.type_id for class_type in CLASSES_TO_LABEL]

def current_captured_frame_num():
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(OUTPUT_FOLDER, 'label_2/')
        print(os.path.abspath(label_path))
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print(num_existing_data_files)
        if num_existing_data_files == 0:
            return 0
        #answer = input(
        #    "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(OUTPUT_FOLDER))
        answer = "A"
        print("There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(OUTPUT_FOLDER))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

def pause_time(prev_save,client):
    now=ceil(time.time())
    if (now-prev_save)>=SAVE_GAP-1:
        vehicles = [x for x in client.get_world().get_actors().filter('vehicle.*')]
        batch = [carla.command.SetAutopilot(x.id, False) for x in vehicles]
        batch+=[carla.command.ApplyVelocity(x.id, carla.Vector3D(0)) for x in vehicles]
        client.apply_batch(batch)
        print("Freezed all cars at " + (time.asctime( time.localtime(now ))))
    return

def restore_ap(client):
    vehicles = [x for x in client.get_world().get_actors().filter('vehicle.*')]
    batch = [carla.command.SetAutopilot(x.id, True) for x in vehicles]
    client.apply_batch(batch)
    print("AP restored")
    return

def saving_time(prev_time):
    now=ceil(time.time())
    if now % SAVE_GAP == 0 and now != prev_time:
        print(now)
        print(prev_time)
        print("Test1 Saving data at " + (time.asctime( time.localtime(now ))))
        return now
    else:
        return None

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.audi.*',
        help='vehicles filter (default: "vehicle.audi.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    args = argparser.parse_args()


    prev_time=0
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    timeout=5.0
    client.set_timeout(timeout)

    world = client.get_world()
    world = client.load_world('Town03')

    try:
        m = world.get_map()
        #start_pose = random.choice(m.get_spawn_points())
        start_pose = carla.Transform(carla.Location(x=-77.8, y=16, z=0))
        #start_pose = carla.Transform(carla.Location(x=-65, y=-3, z=0))
        #start_pose = carla.Transform(carla.Location(x=-97, y=-0.4, z=0))
        #start_pose = carla.Transform(carla.Location(x=-85, y=-11, z=0))
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        '''
        for tl in world.get_actors().filter('traffic.traffic_light*'):
            tl.set_red_time(15)
        '''



        agents=[]
        vehicles_list = []
        walkers_list = []
        all_id = []
        agents=[]

        #random.choice(blueprint_library.filter('vehicle.bmw.*')),

        player_bp=blueprint_library.find('vehicle.tesla.model3')
        intrinsic, cam_rgb_bp, cam_depth_bp, lidar_bp, camera_to_car_transform, lidar_to_car_transform, rgb_transform, depth_transform, lidar_transform = sensor_setting(world)
        
        vehicle = world.spawn_actor(
            cam_rgb_bp,
            start_pose)
        #print ("******************#########")

        

        #car_transform = carla.Transform(carla.Location(x=9, y=-120))
        #vehicle = world.spawn_actor(random.choice(blueprint_library.filter('vehicle.*')), car_transform, attach_to=None)
        vehicle.set_transform(waypoint.transform)
        #txm=carla.Transform(carla.Location(x=-5.5, z=5), carla.Rotation(pitch=-15))
        #vehicle.set_transform(txm)
        #location=vehicle.get_location()
        #location.z+=3
        #vehicle.set_location(location)
        actor_list.append(vehicle)
        
        print(start_pose)
        #vehicle.set_simulate_physics(True)
        #vehicle.set_autopilot(True)

        camera_rgb = world.spawn_actor(
            cam_rgb_bp,
            rgb_transform,
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_depth = world.spawn_actor(
            cam_depth_bp,
            depth_transform,
            attach_to=vehicle)
        actor_list.append(camera_depth)

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=vehicle)
        actor_list.append(lidar)

        #camera_semseg = world.spawn_actor(
        #    blueprint_library.find('sensor.camera.semantic_segmentation'),
        #    carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        #    attach_to=vehicle)
        #actor_list.append(camera_semseg)

        # --------------
        #Spawning NPCs
        # --------------

        spawn_points = m.get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # -----------
        blueprints = world.get_blueprint_library().filter(args.filterv)
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)



        # -------------
        # Spawn Walkers
        # -------------
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invencible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # random max speed
            all_actors[i].set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)


        
        captured_frame_no=current_captured_frame_num()
        print(lidar)
        with CarlaSyncMode(world, lidar, camera_rgb, camera_depth, fps=10) as sync_mode:
        #with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            
            while True:
                if should_quit():
                    return
                clock.tick()
                #print(clock.get_rawtime())
                #print(clock.get_fps())

                # Advance the simulation and wait for the data.
                #snapshot, image_rgb= sync_mode.tick(timeout=2.0)
                snapshot, image_lidar, image_rgb, image_depth = sync_mode.tick(timeout=5.0)
                pc_arr=np.asarray(image_lidar.raw_data)

                #agents=world.get_actors(vehicles_list)
                now=saving_time(prev_time)
                image, datapoints, point_cloud, extrinsic = processing(world,image_rgb, image_depth, image_lidar, intrinsic, vehicle,vehicles_list,camera_to_car_transform,lidar_to_car_transform, (now is not None))
                #image_lidar.save_to_disk('output/%06d.bin' % image_lidar.frame)
                # Choose the next waypoint and update the car location.
                #waypoint = random.choice(waypoint.next(1.5))
                #vehicle.set_transform(waypoint.transform)
                #vehicle.set_autopilot(True)

                #print(vehicle.get_location())
                #image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                pause_time(prev_time,client)

                # Draw the display.
                draw_image(display, image)
                #draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

                #Save Data
                if now is not None and point_cloud.shape[0]>0 and datapoints is not None:
                    save_training_files(vehicle, image_rgb, datapoints, point_cloud,intrinsic,extrinsic,captured_frame_no)
                    captured_frame_no+=1
                    prev_time=now
                    restore_ap(client)
                    
                    #if vehicle.is_at_traffic_light():
                    #    traffic_light = vehicle.get_traffic_light()
                    
                    #    if traffic_light.get_state() == carla.TrafficLightState.Red:
                            # world.hud.notification("Traffic light changed! Good to go!")
                    #        traffic_light.set_state(carla.TrafficLightState.Green)
                    #        print("Traffic changed manually")
                    




    finally:

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controler, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
