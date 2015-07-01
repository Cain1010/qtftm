#include "batchviewwidget.h"
#include "ui_batchviewwidget.h"
#include "batchsurvey.h"
#include "batchdr.h"
#include "batch.h"
#include "batchattenuation.h"
#include "surveyplot.h"
#include "drplot.h"
#include "batchscanplot.h"
#include "batchattnplot.h"
#include <QThread>

BatchViewWidget::BatchViewWidget(BatchManager::BatchType type, int num, double delay, int hpf, double exp, bool rDC, bool pad, QWidget *parent) :
    QWidget(parent), ui(new Ui::BatchViewWidget), d_number(num), d_type(type)
{
    ui->setupUi(this);

    switch(d_type)
    {
    case BatchManager::Survey:
        batchPlot = new SurveyPlot(d_number,this);
        break;
    case BatchManager::DrScan:
        batchPlot = new DrPlot(d_number,this);
        break;
    case BatchManager::Batch:
        batchPlot = new BatchScanPlot(d_number,this);
        break;
    case BatchManager::Attenuation:
        batchPlot = new BatchAttnPlot(d_number,this);
        break;
    default:
        break;
    }


    ui->batchSplitter->insertWidget(d_number,batchPlot);

    ui->delayDoubleSpinBox->blockSignals(true);
    ui->highPassFilterSpinBox->blockSignals(true);
    ui->exponentialFilterDoubleSpinBox->blockSignals(true);

    ui->delayDoubleSpinBox->setValue(delay);
    ui->highPassFilterSpinBox->setValue(hpf);
    ui->exponentialFilterDoubleSpinBox->setValue(exp);
    ui->removeDCCheckBox->setChecked(rDC);
    ui->zeroPadFIDsCheckBox->setChecked(pad);

    ui->analysisWidget->plot()->getDelayBox()->setValue(delay);
    ui->analysisWidget->plot()->getHpfBox()->setValue(hpf);
    ui->analysisWidget->plot()->getExpBox()->setValue(exp);
    ui->analysisWidget->plot()->getRemoveDcBox()->setChecked(rDC);
    ui->analysisWidget->plot()->getPadFidBox()->setChecked(pad);

    connect(ui->printScanButton,&QAbstractButton::clicked,ui->analysisWidget,&AnalysisWidget::print);
    connect(ui->peakListWidget,&PeakListWidget::scanSelected,ui->analysisWidget,&AnalysisWidget::loadScan);
    connect(ui->analysisWidget,&AnalysisWidget::scanChanged,ui->peakListWidget,&PeakListWidget::selectScan);
    connect(ui->analysisWidget,&AnalysisWidget::peakAddRequested,ui->peakListWidget,&PeakListWidget::addUniqueLine);
    connect(ui->reprocessButton,&QAbstractButton::clicked,this,&BatchViewWidget::process);
    connect(this,&BatchViewWidget::checkForMetaDataChanged,ui->analysisWidget,&AnalysisWidget::checkForLoadScanMetaData);
    connect(ui->analysisWidget,&AnalysisWidget::metaDataChanged,this,&BatchViewWidget::metaDataChanged);

    ui->reprocessButton->setEnabled(false);
    ui->printBatchButton->setEnabled(false);
    ui->printScanButton->setEnabled(false);
    ui->analysisWidget->enableSelection(false);

    batchThread = new QThread();

    setAttribute(Qt::WA_DeleteOnClose,true);

}

BatchViewWidget::~BatchViewWidget()
{
    batchThread->quit();
    batchThread->wait();
    delete batchThread;

    delete ui;
}

void BatchViewWidget::process()
{
    if(batchThread->isRunning())
        return;

    ui->printBatchButton->setEnabled(false);
    ui->printScanButton->setEnabled(false);
    ui->reprocessButton->setEnabled(false);
    ui->analysisWidget->loadScan(0);
    ui->analysisWidget->limitRange(0,0);
    ui->analysisWidget->enableSelection(false);


    ui->statusLabel->setText(QString("Processing..."));
    setCursor(Qt::BusyCursor);

    BatchManager *bm;

    switch(d_type)
    {
    case BatchManager::Survey:
        bm = new BatchSurvey(d_number);
        break;
    case BatchManager::DrScan:
        bm = new BatchDR(d_number);
        break;
    case BatchManager::Batch:
        bm = new Batch(d_number);
        break;
    case BatchManager::Attenuation:
        bm = new BatchAttenuation(d_number);
        break;
    case BatchManager::SingleScan:
    default:
        ui->statusLabel->setText(QString("Somehow, an invalid batch type was selected. Please close and try again."));
        unsetCursor();
        return;
    }

    bm->setFtDelay(ui->delayDoubleSpinBox->value());
    bm->setFtHpf(ui->highPassFilterSpinBox->value());
    bm->setFtExp(ui->exponentialFilterDoubleSpinBox->value());
    bm->setRemoveDC(ui->removeDCCheckBox->isChecked());
    bm->setPadFid(ui->zeroPadFIDsCheckBox->isChecked());

    ui->analysisWidget->plot()->getDelayBox()->setValue(ui->delayDoubleSpinBox->value());
    ui->analysisWidget->plot()->getHpfBox()->setValue(ui->highPassFilterSpinBox->value());
    ui->analysisWidget->plot()->getExpBox()->setValue(ui->exponentialFilterDoubleSpinBox->value());
    ui->analysisWidget->plot()->getRemoveDcBox()->setChecked(ui->removeDCCheckBox->isChecked());
    ui->analysisWidget->plot()->getPadFidBox()->setChecked(ui->zeroPadFIDsCheckBox->isChecked());

    QPair<int,int> range = bm->loadScanRange();
    if((range.first < 1 || range.second < 1) && d_type != BatchManager::Attenuation)
    {
        ui->statusLabel->setText(QString("%1 could not be read from disk.").arg(bm->title()));
        unsetCursor();
        return;
    }


    d_firstScan = range.first;
    d_lastScan = range.second;
    setWindowTitle(bm->title());


    connect(batchThread,&QThread::started,bm,&BatchManager::beginBatch);
    connect(bm,&BatchManager::batchComplete,this,&BatchViewWidget::processingComplete);
    connect(bm,&BatchManager::batchComplete,batchThread,&QThread::quit);
    connect(bm,&BatchManager::titleReady,ui->peakListWidget,&PeakListWidget::setTitle);
    connect(bm,&BatchManager::processingComplete,ui->peakListWidget,&PeakListWidget::addScan);
    connect(batchThread,&QThread::finished,bm,&QObject::deleteLater);

    ui->analysisWidget->plot()->clearRanges();
    ui->peakListWidget->clearAll();

    if(d_type == BatchManager::DrScan)
    {
        QList<QPair<double,double> > ranges = dynamic_cast<BatchDR*>(bm)->integrationRanges();
        ui->analysisWidget->plot()->attachIntegrationRanges(ranges);
    }

    bm->moveToThread(batchThread);

    QByteArray state = ui->batchSplitter->saveState();
    delete batchPlot;
    switch(d_type)
    {
    case BatchManager::Survey:
        batchPlot = new SurveyPlot(d_number,this);
        break;
    case BatchManager::DrScan:
        batchPlot = new DrPlot(d_number,this);
        break;
    case BatchManager::Batch:
        batchPlot = new BatchScanPlot(d_number,this);
        break;
    case BatchManager::Attenuation:
        batchPlot = new BatchAttnPlot(d_number,this);
        break;
    default:
        break;
    }
    batchPlot->disableReplotting();

    connect(bm,&BatchManager::plotData,batchPlot,&AbstractBatchPlot::receiveData);
    connect(ui->printBatchButton,&QAbstractButton::clicked,batchPlot,&AbstractBatchPlot::print);
    connect(ui->analysisWidget,&AnalysisWidget::scanChanged,batchPlot,&AbstractBatchPlot::setSelectedZone);
    connect(batchPlot,&AbstractBatchPlot::requestScan,ui->analysisWidget,&AnalysisWidget::loadScan);
    connect(batchPlot,&AbstractBatchPlot::colorChanged,ui->analysisWidget->plot(),&FtPlot::changeColor);

    ui->batchSplitter->insertWidget(1,batchPlot);
    ui->batchSplitter->restoreState(state);


    batchThread->start();

}

void BatchViewWidget::processingComplete(bool failure)
{
    ui->reprocessButton->setEnabled(!failure);
    ui->printBatchButton->setEnabled(!failure);
    ui->printScanButton->setEnabled(!failure);
    ui->analysisWidget->enableSelection(!failure);

    if(failure)
        ui->statusLabel->setText(QString("An error occurred while loading the batch scan. Please close and try again."));
    else
    {
        if(d_firstScan > 0)
        {
            ui->analysisWidget->limitRange(d_firstScan,d_lastScan);
		  ui->analysisWidget->loadScan(d_lastScan);
        }
        else
        {
            ui->analysisWidget->limitRange(0,0);
            ui->printScanButton->setEnabled(false);
        }

        ui->statusLabel->setText(QString("Processing complete."));
    }

    batchPlot->enableReplotting();
    unsetCursor();

}