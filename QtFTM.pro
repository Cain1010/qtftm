#-------------------------------------------------
#
# Project created by QtCreator 2013-05-26T19:28:00
#
#-------------------------------------------------
#This project is only set up to work properly on UNIX, because save paths for files are hard-coded
#It could be easily modified to run on windows; see the Scan::save() and Scan::parseFile() methods
#as well as the BatchManager::writeReport() implementations.
#Before running this program, the directory /home/data must exist, and must be writable to any/all users
#that may run this program! Recommended premissions for directory are 775.
#
#This is the Qt-FTM5 branch, originally to run on FTM1    Jan 2 2015 PRAA
# modified Dec-Jan to run on Qt 5.4/5.4, FTM 2
#Note that #include directories may need to be changed to switch back to FTM 1, especially
#include <qwt6/qwt_plot_canvas.h>, when it ran qwt 6.1, with Qt 5.1 to
#include <qwt_plot_canvas.h>       qwt 6.1.2, which is necessary to run with Qt 5.4
#Running on FTM 2, Jan 2.  also running on FTM 1, Jan 5.  I prefer not to
#have separate branches (for the moment . . . ??)


QT       += core gui network
CONFIG   += qt thread

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets serialport printsupport

TARGET = qtftm
TEMPLATE = app

INCLUDEPATH += "/usr/Qt/5.4/gcc_64/include"
INCLUDEPATH += "/usr/Qt/5.4/gcc_64"


include(wizard.pri)
include(gui.pri)
include(acquisition.pri)
include(hardware.pri)
include(data.pri)
include(settings.pri)

SOURCES += main.cpp

#unix:!macx: LIBS += -L/usr/local/lib64 -lqwt -lgsl -lgslcblas -lm -llabjackusb

unix:!macx: LIBS +=                         -lgsl -lgslcblas -lm -llabjackusb  #-lQtSerialPort
LIBS += -L/usr/local/qwt-6.1.2/lib -lqwt

QMAKE_CXXFLAGS += -std=c++11


RESOURCES += \
    icons.qrc

#Comment the following line and rebuild to enable the DR synthesizer
#DEFINES += CONFIG_NODRSYNTH

#DR synthesizer can be HP8673 or  HP8340 - choose
DEFINES += HP8340

#This line should reflect which spectrometer the code is being compiled for
#DEFINES += QTFTM_FTM2
DEFINES += QTFTM_FTM1

DISTFILES += \
    Notes.txt
