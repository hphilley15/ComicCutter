import sys
import argparse
import tempfile
import os
import glob
from PIL import Image
import logging
import pathlib
import time
import cv2
import numpy as np
import math
import random
import pdf2image

logging.basicConfig()

logger = logging.getLogger( __name__ )
logger.setLevel(logging.DEBUG)

def extract_pages( fname, angle, root_name, out_dir ):
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f'temp_dir {temp_dir}')
        if ( ( fname[-4:] == '.cbz' ) or ( fname[-4:] == '.cbr' ) ):
            os.system( f'7z x -y "{fname}"' )
            time.sleep(3)
            logger.debug( f'Starting to process images' )
            #print( 'dir', [ f for f in pathlib.Path( '.' ).rglob( '*.{jpg,png,jpeg,gif,pdf}' ) ] )
            page = 0

            for fn in pathlib.Path( '.' ).rglob( '*.jpg' ):
                logger.debug( f'processing image {fn}')
                img = Image.open( pathlib.Path( '.') / fn )
                height, width = img.size

                if angle != 0.0:
                    img = img.rotate( angle/math.pi * 180.0 )

                img = img.resize( (height//4, width//4) )
                img = cv2.cvtColor( np.array(img), cv2.COLOR_BGR2RGB)

                if ( page > -1 ):
                    process_image(img, root_name.format(page=page, scene="{scene}"), out_dir)
                page = page + 1
        elif fname[-4:] == '.pdf':
            images = pdf2image.convert_from_path( fname, output_folder = temp_dir)
            logger.debug( f'extract_pages {fname} loaded {len(images)} images')
            page = 0
            for img in images:
                height, width = img.size

                if angle != 0.0:
                    img = img.rotate( angle/math.pi * 180.0 )

                img = img.resize( (height//4, width//4) )
                img = cv2.cvtColor( np.array(img), cv2.COLOR_BGR2RGB)

                if ( page > -1 ):
                    process_image(img, root_name.format(page=page, scene="{scene}"), out_dir)
                page = page + 1



def process_image( img, templ, out_dir ):
    height, width, depth = img.shape
    int_image = cv2.integral( img )

    h_lines = find_horizontal_lines( img, int_image )
    count = 1

    if ( len(h_lines) > 1 ):
        prev = h_lines[0]
        scene = 1
        for li in range( 1,len(h_lines) ):
            current = h_lines[li]
            v_lines = find_vertical_lines( img, prev[3], current[1], int_image )

            if ( len(v_lines) > 1 ):
                prev_x = v_lines[0][2]
                for vli in range( 1, len(v_lines) ):
                    vl = v_lines[vli]
                    x1,y1,x2,y2 = vl

                    crop = Image.fromarray( img[y1:y2, prev_x:x1,::-1] )
                    crop.save( out_dir / templ.format(scene=scene) )

                    prev_x = x2
                    scene = scene + 1


            prev = current


def find_vertical_lines( img, start_y, end_y, int_image = None ):
    height, width, depth = img.shape
    if int_image is None:
        int_image = cv2.integral(img)

    border = np.average( img[start_y:end_y,0:2], axis=(0,1) )

    min_width = None
    prev_r = 0
    y = 0
    vert_lines = []
    start_x = None

    for x in range( width + 1 ):
        if ( x < width ):
            s = extract_sum(x, start_y, x+1, end_y, int_image)
            r = s / ( (end_y - start_y) * (1) )
        else:
            s = 0.0
            r = 0.0

        dist = np.linalg.norm(border - r )
        logger.debug( f'find_vertical_lines checking r {r} {start_x}-{x} against border {border} dist {dist}')
        if dist < 10:
            if (start_x is None):
                start_x = x
                end_x = None
        else:
            end_x = x - 1
            if (start_x is not None) and ( ( min_width is None) or ( end_x - start_x > min_width ) ):
                logger.debug( f'found vertical line {start_x},{start_y},{end_x},{end_y},{min_width}, {r}' )
                vert_lines.append( [ start_x, start_y, end_x, end_y ] )
                min_width = 0.2 * ( end_x - start_x )
                start_x = None
                end_x = None

    vert_lines = supress_vertical_lines( int_image, vert_lines, border, 0.1 * width )

    logger.debug( f'found vertical lines {len(vert_lines)} {type(img)}')
    for x1,y1,x2,y2 in vert_lines:
        cv2.rectangle( img, (x1,y1), (x2,y2), (random.randint(20,25) * 10, random.randint(0,5) * 10, random.randint(0,5) * 10 ), -1 )
    cv2.imshow("Find Vertical Lines", img)
    cv2.waitKey(1)

    return vert_lines

def supress_vertical_lines( int_image, vert_lines, border, min_width ):
    filtered = []
    p_end_x = 0
    p_dist = None

    for li in range( len(vert_lines) ):
        x1,y1,x2,y2 = vert_lines[li]

        s = extract_sum(x1, y1, x2 + 1, y2 + 1, int_image)
        r = s / ((y2 - y1) * (x2 - x1))

        dist = np.linalg.norm(border - r)

        if x1 - p_end_x < min_width:
            if ( p_dist is None ) or (dist < p_dist):
                if ( len(filtered) > 0):
                    xs,ys,xs2,ys2 = filtered.pop()
                    logger.debug( f'supress_vertical lines removing h line {xs}, {ys}, {xs2}, {ys2} and adding {x1}, {y1}, {x2}, {y2}' )
                filtered.append( [x1,y1,x2,y2] )
            else:
                logger.debug(f'supress_vertical lines skipping h line {x1}, {y1}, {x2}, {y2}')
        else:
            filtered.append( [ x1, y1, x2, y2 ] )
        p_dist = dist
        p_end_x = x2

    return filtered

def supress_horizontal_lines( int_image, h_lines, border, min_height ):
    filtered = []
    p_end_y = 0
    p_dist = None

    for li in range( len(h_lines) ):
        x1,y1,x2,y2 = h_lines[li]

        s = extract_sum(x1, y1, x2 + 1, y2 + 1, int_image)
        r = s / ((y2 - y1) * (x2 - x1))

        dist = np.linalg.norm(border - r)

        if y1 - p_end_y < min_height:
            if ( p_dist is None ) or (dist < p_dist):
                if ( len(filtered) > 0):
                    xs,ys,xs2,ys2 = filtered.pop()
                    logger.debug( f'supress_horizontal line removing h line {xs}, {ys}, {xs2}, {ys2} and adding {x1}, {y1}, {x2}, {y2}' )
                filtered.append( [x1,y1,x2,y2] )
            else:
                logger.debug(f'supress_horizontal line skipping h line {x1}, {y1}, {x2}, {y2}')
        else:
            filtered.append( [ x1, y1, x2, y2 ] )
        p_dist = dist
        p_end_y = y2

    return filtered

def threshold_image( img ):
    lower_white = np.min( img[0:4,0:4,:], axis = (0,1) )
    upper_white = np.max( img[0:4,0:4,:], axis = (0,1) )
    logger.debug( f'lower_white {lower_white} upper_white {upper_white} ')
    thresh = cv2.inRange( img, lower_white, upper_white )
    cv2.imshow( 'Threshold', thresh )
    cv2.waitKey(1)
    return thresh

def remove_noise( img ):
    prev = None
    while( (prev is None) or ( np.any(img != prev ) ) ):
        logger.debug( f'morphology open')
        prev = img.copy()
        img = cv2.morphologyEx( prev, cv2.MORPH_OPEN, np.ones( (5,5), np.uint8 ) )

    img = cv2.erode( img, kernel=np.ones( (5,5), np.uint8 ), iterations=3 )

    cv2.imshow("Morphology", img )
    cv2.waitKey(1)

    return img

def extract_edges( img ):
    """ returns four images for right, left, up, and down edges """
    height, width, depth = img.shape
    dtype = img.dtype

    out = np.zeros( (height, width, 1), dtype=dtype )

    #left
    for y in range(height):
        white_count = 0
        non_white_count = 0

        for x in range(width+1):
                if x < width:
                    pix = img[y,x,1] # Just use the green channel
                else:
                    pix = 0
                if pix > 200:
                    non_white_count = 0
                    white_count = white_count + 1
                elif pix < 200:
                    if white_count > 5:
                        if x < width:
                            out[y,x,:] = 255
                        if ( x - white_count > 0 ):
                            out[y,x-white_count,:] = 255
                    white_count = 0
                    non_white_count = non_white_count + 1
    return out

def extract_sum( x1, y1, x2, y2, int_image ):
    return int_image[y2,x2] - int_image[y1,x2] - int_image[y2,x1] + int_image[y1,x1]

def find_horizontal_lines( img, int_image = None ):
    height, width, depth = img.shape
    if ( int_image is None ):
        int_image = cv2.integral( img )

    border = np.average( img[0:4,0:4,:], axis=(0,1) )

    min_height = None
    prev_r = 0
    y = 0
    horiz_lines = []
    start_y = None

    for y in range( height + 1 ):
        start_x = 3
        end_x = width - 3

        if ( y < height ):
            s = extract_sum(start_x, y, end_x, y+1, int_image)
            r = s / ( (end_x - start_x) * (1) )
        else:
            s = 0.0
            r = 0.0

        dist = np.linalg.norm(border - r)
        if dist < 10:
            if (start_y is None):
                start_y = y
                end_y = None
        else:
            end_y = y - 1
            if (start_y is not None) and ( ( min_height is None) or ( end_y - start_y > min_height ) ):
                horiz_lines.append( [ start_x, start_y, end_x, end_y ] )
                min_height = 0.5 * ( end_y - start_y )
                start_y = None
                end_y = None

    horiz_lines = supress_horizontal_lines( int_image, horiz_lines, border, 0.1 * height )
    logger.debug( f'found horizontal lines {len(horiz_lines)} {type(img)}')

    if ( (len(horiz_lines) > 0 ) and ( height - horiz_lines[-1][3] > 0.2*height ) ):
        horiz_lines.append( ( 0, height - 1, width, height ) )

    for x1,y1,x2,y2 in horiz_lines:
        cv2.rectangle( img, (x1,y1), (x2,y2), (random.randint(0,5) * 10, random.randint(0,5) * 10, random.randint(0,5) * 10 ), -1 )
    cv2.imshow("Find Horizontal Lines", img)
    cv2.waitKey(1)

    return horiz_lines

def xxx_find_lines( img ):
    height, width, depth = img.shape

    int_img = cv2.Integral( img, )

    # thresh = threshold_image( img )
    # noise = remove_noise( thresh )

    #cv2.imshow('Original', img)

    #gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    # kernel_size = 7
    # blur = cv2.GaussianBlur( thresh, (kernel_size, kernel_size), 0)
    low_threshold = 80
    high_threshold = 150
    edges = cv2.Canny( noise, low_threshold, high_threshold)
    #edges = extract_edges( img )
    cv2.imshow( 'Canny Edge Detection', edges )
    cv2.waitKey( 1 )

    rho = 5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 45  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = min(height, width)//20  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.zeros(img.shape, dtype=img.dtype)  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    print(lines)
    points = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = math.atan2( max(x1,x2) - min(x1,x2), max(y1,y2) - min(y1,y2) )
                logger.debug( f'angle {angle/math.pi * 180.0:5.2f} deg.')
                if ( abs(angle - 0.0) < 25.0/180.0 * math.pi ) or ( abs(angle - math.pi/2) < 25.0/180.0 * math.pi ):
                    points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 1, 0), 5)

    lines_edges = cv2.addWeighted(img, 0.4, line_image, 1, 0)
    print(lines_edges.shape)
    cv2.imshow( 'Sample', lines_edges )
    cv2.waitKey(1);


def main( argv = None ):
    if not argv:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser( description='Extract images from a cbr/cbz file' )
    parser.add_argument( 'files', nargs='+', help='list of comic books (cbr/cbz/pdf)' )
    parser.add_argument( '--rotate', default=0, type=float, help='rotate page by angle angle' )
    parser.add_argument( '--template', default="p{page}_{scene}.png")
    parser.add_argument( '--out_dir', default='./out', help='output_directory' )

    args = parser.parse_args( argv )
    args.rotate = args.rotate/180.0 * math.pi

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for fn in args.files:
        print( f'Processing files {fn} {args.rotate/math.pi * 180.0}' )
        extract_pages( fn, args.rotate, args.template, out_dir )

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#    main( [ "C:/Users/hphil/Desktop/Lucky Luke 18 - Der singende Draht.cbr", '--template', 'll_18_p{page}_s{scene}.png' ] )
#    main( [ '--rotate', "-90", "D:/Downloads/BANDE DESSINEE Tintin - Tintin et le Lac aux requins.pdf" ] )
    main([ "D:/Downloads/BANDE DESSINEE michel vaillant T15 - Le cirque infernal.pdf", '--template', 'mv_15_[{page}_s{scene}.png', '--out_dir', './out'] )
#    main(["D:/Downloads/BANDE DESSINEE Blueberry - 06 - L Homme A L Etoile D Argent.pdf"])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
