from VesicleViewer_ui import Ui_VesicleViewer
from moviereader import MovieReader

import sys
import tables
import os
import numpy as np
import copy
import pandas as pd
from functools import partial

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QFont, QPen, QPolygonF, QColor


def setChildrenFocusPolicy(obj, policy):
    
    # recursively change the focus policy of all the objects in the widgets
    def recursiveSetChildFocusPolicy(parentQWidget):
        for childQWidget in parentQWidget.findChildren(QtWidgets.QWidget):
            childQWidget.setFocusPolicy(policy)
            recursiveSetChildFocusPolicy(childQWidget)
    recursiveSetChildFocusPolicy(obj)

class LineEditDragDrop():
    def __init__(self, main_obj, update_fun, test_file_fun):
        self.update_fun = update_fun
        self.test_file_fun = test_file_fun

        self.main_obj = main_obj
        if isinstance(self.main_obj, QtWidgets.QLineEdit):
            self.line_edit_obj = self.main_obj
        else:
            self.line_edit_obj = self.main_obj.lineEdit()
            
        self.main_obj.setAcceptDrops(True)
        self.main_obj.dragEnterEvent = self.dragEnterEvent
        self.main_obj.dropEvent = self.dropEvent
        self.line_edit_obj.returnPressed.connect(self.returnPressedFun)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            vfilename = url.toLocalFile()
            if self.test_file_fun(vfilename):
                self.update_fun(vfilename)

    def returnPressedFun(self):
        vfilename = self.line_edit_obj.text()
        if self.test_file_fun(vfilename):
            self.update_fun(vfilename)

class ViewsWithZoom():

    def __init__(self, view):
        self._view = view
        self._scene = QtWidgets.QGraphicsScene(self._view)
        self._view.setScene(self._scene)
        self._canvas = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._canvas)

        self._zoom = 0
        self._view.wheelEvent = self.zoomWheelEvent

    # zoom wheel
    def zoomWheelEvent(self, event):
        if not self._canvas.pixmap().isNull():
            numPixels = event.pixelDelta()
            numDegrees = event.angleDelta() / 8

            delta = numPixels if not numPixels.isNull() else numDegrees
            self.zoom(delta.y())

    def zoom(self, zoom_direction):
        if zoom_direction > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        if self._zoom > 0:
            self._view.scale(factor, factor)
        elif self._zoom == 0:
            self.zoomFitInView()
        else:
            self._zoom = 0

    def zoomFitInView(self):
        rect = QtCore.QRectF(self._canvas.pixmap().rect())
        if not rect.isNull():
            unity = self._view.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self._view.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self._view.viewport().rect()
            scenerect = self._view.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self._view.scale(factor, factor)
            self._view.centerOn(rect.center())
            self._zoom = 0

    def cleanCanvas(self):
        self._canvas.setPixmap(QtGui.QPixmap())

    def setPixmap(self, frame_qimg=None):
        if frame_qimg is None:
            return

        pixmap = QtGui.QPixmap.fromImage(frame_qimg)
        self._canvas.setPixmap(pixmap)

class SimplePlayer(QtWidgets.QMainWindow):
    def __init__(self, ui):
        super().__init__()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.getNextImage)
        self.ui = ui
        self.isPlay = False
        self.image_group = None
        self.frame_number = None

    def keyPressEvent(self, event):
        #HOT KEYS
        key = event.key()

        # Duplicate the frame step size (speed) when pressed  > or .:
        if key == Qt.Key_Greater or key == Qt.Key_Period:
            self.frame_step *= 2
            self.ui.spinBox_step.setValue(self.frame_step)
            

        # Half the frame step size (speed) when pressed: < or ,
        elif key == Qt.Key_Less or key == Qt.Key_Comma:
            self.frame_step //= 2
            if self.frame_step < 1:
                self.frame_step = 1
            self.ui.spinBox_step.setValue(self.frame_step)
            

        # Move backwards when  are pressed
        elif key == Qt.Key_Left:
            self.frame_number -= self.frame_step
            if self.frame_number < 0:
                self.frame_number = 0
            self.ui.spinBox_frame.setValue(self.frame_number)
            

        # Move forward when  are pressed
        elif key == Qt.Key_Right:
            self.frame_number += self.frame_step
            if self.frame_number >= self.tot_frames:
                self.frame_number = self.tot_frames - 1
            self.ui.spinBox_frame.setValue(self.frame_number)
            
        #super().keyPressEvent(event)

    def playVideo(self):
        if self.image_group is None:
            return
        if not self.isPlay:
            self.startPlay()
        else:
            self.stopPlay()

    def startPlay(self):
        self.timer.start(round(1000 / self.fps))
        self.isPlay = True
        self.ui.playButton.setText('Stop')
        self.ui.doubleSpinBox_fps.setEnabled(False)

    def stopPlay(self):
        self.timer.stop()
        self.isPlay = False
        self.ui.playButton.setText('Play')
        self.ui.doubleSpinBox_fps.setEnabled(True)

    # Function to get the new valid frame during video play
    def getNextImage(self):
        self.frame_number += self.frame_step
        if self.frame_number >= self.tot_frames:
            self.frame_number = self.tot_frames - 1
            self.stopPlay()
        self.ui.spinBox_frame.setValue(self.frame_number)

    @property
    def fps(self):
        return self.ui.doubleSpinBox_fps.value()
    @fps.setter
    def fps(self, value):
        return self.ui.doubleSpinBox_fps.setValue(value)

    @property
    def frame_step(self):
        return self.ui.spinBox_step.value()

    @frame_step.setter
    def frame_step(self, value):
        return self.ui.spinBox_step.setValue(value)


class IntensitySlider():
    _slider_n_steps = 200

    def __init__(self, ui):
        self.spinBox_int_low = ui.spinBox_int_low
        self.spinBox_int_high = ui.spinBox_int_high
        self.horizontalSlider_int_low = ui.horizontalSlider_int_low
        self.horizontalSlider_int_high = ui.horizontalSlider_int_high

        self.horizontalSlider_int_low.setMinimum(0)
        self.horizontalSlider_int_high.setMinimum(0)
        self.horizontalSlider_int_low.setMaximum(self._slider_n_steps)
        self.horizontalSlider_int_high.setMaximum(self._slider_n_steps)
        
        self.link_spinbox_to_slide(self.spinBox_int_low, self.horizontalSlider_int_low, self._slider_n_steps)
        self.link_spinbox_to_slide(self.spinBox_int_high, self.horizontalSlider_int_high, self._slider_n_steps)

        self.bound_widgets(self.spinBox_int_low, self.spinBox_int_high)
        self.bound_widgets(self.horizontalSlider_int_low, self.horizontalSlider_int_high)
        
        self.set_range(0, 255, step = 1)
        
    def link_spinbox_to_slide(self, spinbox, slider, slider_n_steps):
        def _spinbox_changed(spinbox_val):
            bot = spinbox.minimum()
            top = spinbox.maximum()


            ll = max((top-bot), 1)
            slider_val = (spinbox_val-bot)/ll*slider_n_steps


            slider.blockSignals(True)
            slider.setValue(slider_val)
            slider.blockSignals(False)

        def _slider_changed(slider_val):
            bot = spinbox.minimum()
            top = spinbox.maximum()
            spinbox_val = slider_val/slider_n_steps*(top-bot) + bot

            spinbox.blockSignals(True)
            spinbox.setValue(spinbox_val)
            spinbox.blockSignals(False)

        spinbox.valueChanged.connect(_spinbox_changed)
        slider.valueChanged.connect(_slider_changed)

    def bound_widgets(self, widget_low, widget_high):
        def _bound_values(val=None):
            low_val = widget_low.value()
            hi_val = widget_high.value()
            
            if low_val >= hi_val:
                widget_low.setValue(hi_val - self._step)
            elif hi_val <= low_val:
                widget_high.setValue(low_val + self._step)
        
        widget_high.valueChanged.connect(_bound_values)
        widget_low.valueChanged.connect(_bound_values)
        



    def set_range(self, min_val, max_val, step):
        self._step = step

        self.spinBox_int_low.setSingleStep(step)
        self.spinBox_int_high.setSingleStep(step)
        
        self.spinBox_int_low.setMaximum(max_val)
        self.spinBox_int_high.setMaximum(max_val)
        
        self.spinBox_int_low.setMinimum(min_val)
        self.spinBox_int_high.setMinimum(min_val)

        self.spinBox_int_high.setValue(max_val)
        self.spinBox_int_low.setValue(min_val)

    def set_enable(self, val):
        self.spinBox_int_low.setEnabled(val)
        self.spinBox_int_high.setEnabled(val)
        self.horizontalSlider_int_low.setEnabled(val)
        self.horizontalSlider_int_high.setEnabled(val)

    def get_range(self):
        bot = self.spinBox_int_low.value()
        top = self.spinBox_int_high.value()
        return bot, top

    def log_range(self, is_log):
        cur_bot = self.spinBox_int_low.value()
        cur_top = self.spinBox_int_high.value() 
        bot, top = self.spinBox_int_low.minimum(), self.spinBox_int_high.maximum() 
        
        self.spinBox_int_low.blockSignals(True)
        self.spinBox_int_high.blockSignals(True)
        if is_log:
            bot = np.log(bot + 1) 
            top = np.log(top)

            cur_bot = np.log(cur_bot + 1)
            cur_top = np.log(cur_top)

            self.set_range(bot, top, step = 0.01)
            
            
        else:
            bot = np.exp(bot)-1 
            top = np.exp(top)

            cur_bot = np.exp(cur_bot)-1
            cur_top = np.exp(cur_top)

            self.set_range(bot, top, step = 1)

        cur_bot = max(bot, cur_bot)
        cur_top = min(top, cur_top)


        self.spinBox_int_low.blockSignals(False)
        self.spinBox_int_high.blockSignals(False)

        self.spinBox_int_low.setValue(cur_bot)
        self.spinBox_int_high.setValue(cur_top)



class HDF5Reader():
    def __init__(self):
        super().__init__()

    def _updateVideoFile(self, vfilename):

        try:
            self.fid = tables.File(vfilename, 'r')
        except (IOError, tables.exceptions.HDF5ExtError):
            self.fid = None
            self.image_group = None
            QtWidgets.QMessageBox.critical(
                self,
                '',
                "The selected file is not a valid .hdf5. Please select a valid file",
                QtWidgets.QMessageBox.Ok)
            return

        self.updateGroupNames()


    def updateGroupNames(self):
        if self.fid is None:
            return

        valid_groups = []
        for group in self.fid.walk_groups("/"):
            print(group)
            for array in self.fid.list_nodes(group, classname='Array'):
                if array.ndim == 3:
                    valid_groups.append(array._v_pathname)


        if not len(valid_groups):
            QtWidgets.QMessageBox.critical(
                self,
                '',
                "No valid video groups were found. Dataset with three dimensions. Closing file.",
                QtWidgets.QMessageBox.Ok)
            self.fid.close()
            self.image_group = None
            self.mainImage.cleanCanvas()

            return

        self.ui.comboBox_h5path.clear()

        imask = 0
        for ii, kk in enumerate(valid_groups):
            self.ui.comboBox_h5path.addItem(kk)
            if kk == '/mask':
                imask = ii

        self.getImGroup(imask)
        self.updateImage()

    def getImGroup(self, index):
        txt = self.ui.comboBox_h5path.itemText(index)
        self.ui.comboBox_h5path.setCurrentIndex(index)
        self.updateImGroup(txt)

    # read a valid groupset from the hdf5
    def updateImGroup(self, h5path):
        #self.h5path = self.ui.comboBox_h5path.text()
        if h5path not in self.fid:
            self.mainImage.cleanCanvas()
            QtWidgets.QMessageBox.critical(
                self,
                'The groupset path does not exist',
                "The groupset path does not exists. You must specify a valid groupset path",
                QtWidgets.QMessageBox.Ok)
            self.image_group == None
            return

        self.h5path = h5path
        self.image_group = self.fid.get_node(h5path)
        if len(self.image_group.shape) != 3:
            self.mainImage.cleanCanvas()
            QtWidgets.QMessageBox.critical(
                self,
                'Invalid groupset',
                "Invalid groupset. The groupset must have three dimensions",
                QtWidgets.QMessageBox.Ok)
            self.image_group == None
            return 

        

        self.tot_frames = self.image_group.shape[0]
        self.image_height = self.image_group.shape[1]
        self.image_width = self.image_group.shape[2]

        #just control that the current frame will not overflow the array
        if self.frame_number > self.tot_frames - 1:
             self.frame_number = self.tot_frames - 1
        

class MovieReaderF(MovieReader):
    def __init__(self, *args, **argkws):
        super().__init__(*args, **argkws)
    def __getitem__(self, index):
        return super().__getitem__(index)[1]

class JurijMovieReader():
    def __init__(self):
        super().__init__()

    def _updateVideoFile(self, vfilename):
        self.image_group = MovieReaderF(vfilename)
        self.tot_frames = len(self.image_group)
        self.image_height = self.height
        self.image_width = self.width
        self.frame_number = 0

class ContourPlotter():
    def __init__(self):
        super().__init__()
        self.frames_data = None
        self.coords_data = None
        self.cnt_per_frame = None
        self.coord_per_cnt = None
        self.traj_colors = {}

    def load(self, fname):
        self.traj_colors = {}
        with pd.HDFStore(fname, 'r') as fid:
            if '/contours_data' in fid:
                self.frames_data = fid['/contours_data']
                self.coords_data = fid['/contours_coordinates']
            
                self.cnt_per_frame = self.frames_data.groupby('frame_number').groups
                self.coord_per_cnt = self.coords_data.groupby('contour_id').groups
                
            else:
                self.frames_data = None
                self.coords_data = None
                self.cnt_per_frame = None
                self.coord_per_cnt = None

    def plot(self, frame_number, image):
        if self.frames_data is None:
            return

        if not frame_number in self.cnt_per_frame:
            return

        painter = QPainter()
        painter.begin(image)

        penwidth = max(1, max(image.height(), image.width()) // 800)
        pen = QPen()
        pen.setWidth(penwidth)
        
        frame_data = self.frames_data.loc[self.cnt_per_frame[frame_number]]

        for _, row in frame_data.iterrows():
            interface_id = row['interface_id']
            cnt_id = row['contour_id']

            if not cnt_id in self.coord_per_cnt:
                continue
            
            coords = self.coords_data.loc[self.coord_per_cnt[cnt_id]]
            xx = coords['X']
            yy = coords['Y']

            if not interface_id in self.traj_colors:
                self.traj_colors[interface_id] = QColor(*np.random.randint(50, 230, 3))
            col = self.traj_colors[interface_id]

            p = QPolygonF()
            for x,y in zip(xx, yy):
                p.append(QPointF(x,y))

            pen.setColor(col)
            painter.setPen(pen)
            painter.drawPolyline(p)
        
        painter.end()


class VideoPlayerGUI(SimplePlayer, HDF5Reader, JurijMovieReader):

    def __init__(self, ui=None):
        if ui is None:
            ui = Ui_VesicleViewer()
        
        super().__init__(ui)

        # Set up the user interface from Designer.
        self.ui.setupUi(self)

        self.vfilename = None
        self.fid = None
        self.image_group = None
        self.isPlay = False
        self.videos_dir = ''
        self.h5path = None
        self.frame_raw = None
        self.frame_img = None
        self.frame_qimg = None

        self.ctn_plotter = ContourPlotter()

        #default expected groups in the hdf5
        self.ui.comboBox_h5path.setItemText(0, "/mask")
        self.ui.comboBox_h5path.setItemText(1, "/full_data")

        self.ui.pushButton_video.clicked.connect(self.getVideoFile)
        self.ui.playButton.clicked.connect(self.playVideo)

        # set scroller
        sld_pressed = partial(self.ui.imageSlider.setCursor, QtCore.Qt.ClosedHandCursor)
        sld_released = partial(self.ui.imageSlider.setCursor, QtCore.Qt.OpenHandCursor)
        
        self.ui.imageSlider.sliderPressed.connect(sld_pressed)
        self.ui.imageSlider.sliderReleased.connect(sld_released)
        self.ui.imageSlider.valueChanged.connect(self.ui.spinBox_frame.setValue)
        #eliminate ticks, they will be a problem since I make the maximum size of the slider tot_frames
        self.ui.imageSlider.setTickPosition(QtWidgets.QSlider.NoTicks)

        #%%
        self.ui.spinBox_frame.valueChanged.connect(self.updateFrameNumber)
        self.ui.comboBox_h5path.activated.connect(self.getImGroup)
        self.ui.pushButton_h5groups.clicked.connect(self.updateGroupNames)

        self.intensity_slider = IntensitySlider(self.ui)
        self.ui.horizontalSlider_int_low.sliderReleased.connect(self.updateImage)
        self.ui.horizontalSlider_int_high.sliderReleased.connect(self.updateImage)
        self.ui.pushButton_autoscale.clicked.connect(self._autoscale)

        # setup image view as a zoom
        self.mainImage = ViewsWithZoom(self.ui.mainGraphicsView)

        # let drag and drop a file into the video file line edit
        LineEditDragDrop(
            self.ui.lineEdit_video,
            self.updateVideoFile,
            os.path.isfile)

        # make sure the childrenfocus policy is none in order to be able to use
        # the arrow keys
        setChildrenFocusPolicy(self, QtCore.Qt.ClickFocus)
    
    def keyPressEvent(self, event):
        #HOT KEYS

        if self.image_group is None:
            # break no file open, nothing to do here
            return

        key = event.key()
        if key == Qt.Key_Minus:
            self.mainImage.zoom(-1)
        elif key == Qt.Key_Plus:
            self.mainImage.zoom(1)

        super().keyPressEvent(event)

    # frame spin box
    def updateFrameNumber(self):
        self.frame_number = self.ui.spinBox_frame.value()
        self.ui.imageSlider.setValue(self.frame_number)
        self.updateImage()

    # update image: get the next frame_number, and resize it to fix in the GUI
    # area
    def updateImage(self):
        self.readCurrentFrame()
        self.ctn_plotter.plot(self.frame_number, self.frame_qimg)
        self.mainImage.setPixmap(self.frame_qimg)

    def readCurrentFrame(self):
        if self.image_group is None:
            self.frame_qimg = None
            return
        self.frame_raw = self.image_group[self.frame_number]
        if self.ui.checkBox_is_log.isChecked():
            self.frame_raw = np.log(self.frame_raw + 1)
        self._normalizeImage()
        

    def _normalizeImage(self):
        if self.frame_raw is None:
            return 

        dd = self.ui.mainGraphicsView.size()
        self.label_height = dd.height()
        self.label_width = dd.width()

        bot, top = self.intensity_slider.get_range()

        self.frame_img = (self.frame_raw.astype(np.float) - bot) / (top - bot)
        self.frame_img = np.clip(np.round(self.frame_img* 255.), 0, 255)
        self.frame_img = self.frame_img.astype(np.uint8)

        self.frame_qimg = self._convert2Qimg(self.frame_img)


    def _autoscale(self):
        if self.frame_raw is None:
            return

        bot = np.min(self.frame_raw)
        top = np.max(self.frame_raw)

        self.intensity_slider.spinBox_int_low.setValue(bot) 
        self.intensity_slider.spinBox_int_high.setValue(top) 
        self.updateImage()

    def _convert2Qimg(self, img):
        qimg = QtGui.QImage(
            img.data,
            img.shape[1],
            img.shape[0],
            img.strides[0],
            QtGui.QImage.Format_Indexed8)
        qimg = qimg.convertToFormat(
            QtGui.QImage.Format_RGB32, QtCore.Qt.AutoColor)

        return qimg

    # file dialog to the the hdf5 file
    def getVideoFile(self):
        vfilename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Find HDF5 video file", self.videos_dir, "HDF5 files (*.hdf5);; All files (*)")

        self.updateVideoFile(vfilename)

    def setFileName(self, filename):
        self.filename = filename
        self.ui.lineEdit.setText(filename)

    def resizeEvent(self, event):
        if self.image_group is not None:
            self.updateImage()
            self.mainImage.zoomFitInView()


    def closeEvent(self, event):
        if self.fid is not None:
            self.fid.close()
        super().closeEvent(event)


    def updateVideoFile(self, vfilename):

        # close the if there was another file opened before.
        if self.fid is not None:
            self.fid.close()
        
        self.mainImage.cleanCanvas()
        self.fid = None
        self.image_group = None
        self.h5path = None
        self.frame_number = 0
        self.tot_frames = -1

        self.vfilename = vfilename
        self.ui.lineEdit_video.setText(self.vfilename)
        self.videos_dir = self.vfilename.rpartition(os.sep)[0] + os.sep

        if self.vfilename.endswith('.hdf5'):
            self.ui.comboBox_h5path.setEnabled(True)
            self.ui.pushButton_h5groups.setEnabled(True)
            HDF5Reader._updateVideoFile(self, self.vfilename)

        elif self.vfilename.endswith('.movie'):
            self.ui.comboBox_h5path.setEnabled(False)
            self.ui.pushButton_h5groups.setEnabled(False)
            JurijMovieReader._updateVideoFile(self, self.vfilename)

        self.ctn_plotter.load(self.vfilename)

        self.ui.spinBox_frame.setMaximum(self.tot_frames - 1)
        self.ui.imageSlider.setMaximum(self.tot_frames - 1)


        self.ui.spinBox_frame.setValue(self.frame_number)
        
        self.mainImage.zoomFitInView()


        if self.image_group.dtype == np.uint8:
            self.intensity_slider.set_range(0, 255, 1)
        elif self.image_group.dtype == np.uint16:
            self.intensity_slider.set_range(0, 2**16-1, 1)
            
        else:
            #TODO Implement the intensity slider for non unit8 data. For the moment I deactivate this.
            self.intensity_slider.set_enable(False)

        img = self.image_group[self.frame_number]

        self.ui.checkBox_is_log.toggled.connect(self.intensity_slider.log_range)
        self.ui.checkBox_is_log.toggled.connect(self.updateImage)
        
        self.updateImage()
        self._autoscale()

if __name__ == '__main__':
    print('hello!!')
    app = QtWidgets.QApplication(sys.argv)

    ui = VideoPlayerGUI()
    ui.show()

    sys.exit(app.exec_())
