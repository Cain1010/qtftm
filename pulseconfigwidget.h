#ifndef PULSECONFIGWIDGET_H
#define PULSECONFIGWIDGET_H

#include <QWidget>
#include <QList>

#include "pulsegenconfig.h"
#include "pulsegenerator.h"

class QLabel;
class QDoubleSpinBox;
class QPushButton;
class QToolButton;
class QLineEdit;

namespace Ui {
class PulseConfigWidget;
}

class PulseConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PulseConfigWidget(QWidget *parent = 0);
    ~PulseConfigWidget();

    struct ChWidgets {
        QLabel *label;
        QDoubleSpinBox *delayBox;
        QDoubleSpinBox *widthBox;
        QPushButton *onButton;
        QToolButton *cfgButton;
        QLineEdit *nameEdit;
        QPushButton *levelButton;
        QDoubleSpinBox *delayStepBox;
        QDoubleSpinBox *widthStepBox;
    };

    PulseGenConfig getConfig();

    void makeInternalConnections();


signals:
    void changeSetting(int,QtFTM::PulseSetting,QVariant);
    void changeRepRate(double);

public slots:
    void launchChannelConfig(int ch);
    void newSetting(int index,QtFTM::PulseSetting s,QVariant val);
    void newConfig(const PulseGenConfig c);
    void newRepRate(double r);

private:
    Ui::PulseConfigWidget *ui;

    QList<ChWidgets> d_widgetList;


};

#endif // PULSECONFIGWIDGET_H
