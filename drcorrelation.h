#ifndef DRCORRELATION_H
#define DRCORRELATION_H

#define QTFTM_DRCORR_THRESHOLD 0.5

#include "batchmanager.h"

class AutoFitWidget;

class DrCorrelation : public BatchManager
{
	Q_OBJECT
public:
	explicit DrCorrelation(QList<QPair<Scan,bool>> templateList, AbstractFitter *ftr = new NoFitter());
    explicit DrCorrelation(int num, AbstractFitter *ftr = new NoFitter());
	~DrCorrelation();

	// BatchManager interface
protected:
	void writeReport();
    void advanceBatch(const Scan s);
	void processScan(Scan s);
	Scan prepareNextScan();
	bool isBatchComplete();

private:
    bool d_thisScanIsRef, d_processScanIsCal, d_processScanIsRef;
	double d_currentRefMax;
	QList<QPair<Scan,bool>> d_scanList;
	QVector<QPointF> d_drData, d_calData;
	QStringList d_saveData;

	QList<bool> d_loadCalList, d_loadRefList;
	int d_loadIndex;
};

#endif // DRCORRELATION_H
