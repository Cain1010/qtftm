SOURCES += mainwindow.cpp \
    ftplot.cpp \
    pulseplot.cpp \
    analysiswidget.cpp \
    analysisplot.cpp \
    singlescanwidget.cpp \
    singlescandialog.cpp \
    batchplot.cpp \
    loadbatchdialog.cpp \
    batchviewwidget.cpp \
    autofitwidget.cpp \
    peaklistwidget.cpp \
    $$PWD/led.cpp \
    $$PWD/zoompanplot.cpp \
    abstractbatchplot.cpp \
    $$PWD/surveyplot.cpp \
    $$PWD/drplot.cpp \
    $$PWD/batchscanplot.cpp \
    $$PWD/batchattnplot.cpp

HEADERS += mainwindow.h \
    settingsdialog.h \
    ftplot.h \
    pulseplot.h \
    analysiswidget.h \
    analysisplot.h \
    singlescanwidget.h \
    singlescandialog.h \
    batchplot.h \
    loadbatchdialog.h \
    batchviewwidget.h \
    autofitwidget.h \
    peaklistwidget.h \
    $$PWD/led.h \
    $$PWD/zoompanplot.h \
    abstractbatchplot.h \
    $$PWD/surveyplot.h \
    $$PWD/drplot.h \
    $$PWD/batchscanplot.h \
    $$PWD/batchattnplot.h


FORMS    += mainwindow.ui \
    analysiswidget.ui \
    singlescanwidget.ui \
    singlescandialog.ui \
    loadbatchdialog.ui \
    batchviewwidget.ui \
    autofitwidget.ui \
    peaklistwidget.ui

OTHER_FILES += \
    led.qml