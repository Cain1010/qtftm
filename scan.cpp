#include "scan.h"
#include <QSharedData>
#include <QSettings>
#include <QApplication>
#include <math.h>
#include <QDir>
#include <QDateTime>
#include <QTextStream>

/*!
 \brief Data storage for Scan

 Self-explanatory data storage.

*/
class ScanData : public QSharedData {
public:
	ScanData() : number(-1), ts(QDateTime::currentDateTime()), ftFreq(-1.0), ftAtten(-1), drFreq(-1.0), drPower(-100.0), pressure(-1.0),
        gasNames(QStringList()), gasFlows(QList<double>()), repRate(0.0), pulseConfig(QList<PulseGenerator::PulseChannelConfiguration>()),
	   targetShots(0), completedShots(0), fid(Fid()), initialized(false), saved(false), aborted(false), dummy(false), skipTune(false),
       tuningVoltage(-1), tuningVoltageTakenWithScan(true), scansSinceTuningVoltageTaken(0), cavityVoltage(-1), protectionDelayTime (-1),
       scopeDelayTime(-1), dipoleMoment(0.0), magnet(false) {}
	ScanData(const ScanData &other) :
		QSharedData(other), number(other.number), ts(other.ts), ftFreq(other.ftFreq), ftAtten(other.ftAtten), drFreq(other.drFreq),
        drPower(other.drPower), pressure(other.pressure), gasNames(other.gasNames), gasFlows(other.gasFlows), repRate(other.repRate),
		pulseConfig(other.pulseConfig), targetShots(other.targetShots), completedShots(other.completedShots),
        fid(other.fid), initialized(other.initialized), saved(other.saved), aborted(other.aborted), dummy(other.dummy),
        skipTune(other.skipTune), tuningVoltage(other.tuningVoltage), tuningVoltageTakenWithScan(other.tuningVoltageTakenWithScan),
        scansSinceTuningVoltageTaken(other.scansSinceTuningVoltageTaken), cavityVoltage(other.cavityVoltage),
        protectionDelayTime(other.protectionDelayTime), scopeDelayTime(other.scopeDelayTime), dipoleMoment(other.dipoleMoment),
        magnet(other.magnet) {}
	~ScanData() {}

	int number;
	QDateTime ts;

	double ftFreq;
	int ftAtten;
	double drFreq;
	double drPower;

	double pressure;
	QStringList gasNames;
	QList<double> gasFlows;

    double repRate;
	QList<PulseGenerator::PulseChannelConfiguration> pulseConfig;

	int targetShots;
	int completedShots;

	Fid fid;
	bool initialized;
	bool saved;
	bool aborted;
	bool dummy;
	bool skipTune;

    int tuningVoltage;
    bool tuningVoltageTakenWithScan;
    int  scansSinceTuningVoltageTaken;
    int cavityVoltage;

    int protectionDelayTime;
    int scopeDelayTime;

    double dipoleMoment;
    bool magnet;


};

Scan::Scan() : data(new ScanData)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QStringList gasNames;
    gasNames.append(s.value(QString("gas1Name"),QString("")).toString());
    gasNames.append(s.value(QString("gas2Name"),QString("")).toString());
    gasNames.append(s.value(QString("gas3Name"),QString("")).toString());
    gasNames.append(s.value(QString("gas4Name"),QString("")).toString());
    setGasNames(gasNames);
    setRepRate(s.value(QString("pulseGenerator/repRate"),6.0).toDouble());
}

Scan::Scan(int num) : data(new ScanData)
{
	parseFile(num);
}

Scan::Scan(const Scan &rhs) : data(rhs.data)
{
}

Scan &Scan::operator=(const Scan &rhs)
{
	if (this != &rhs)
		data.operator=(rhs.data);
	return *this;
}

bool Scan::operator ==(const Scan &other) const
{
	return data == other.data;
}

Scan::~Scan()
{
}

int Scan::number() const
{
	return data->number;
}

QDateTime Scan::timeStamp() const
{
	return data->ts;
}

Fid Scan::fid() const
{
	return data->fid;
}

double Scan::ftFreq() const
{
	return data->ftFreq;
}

double Scan::drFreq() const
{
	return data->drFreq;
}

int Scan::attenuation() const
{
	return data->ftAtten;
}

double Scan::dipoleMoment() const
{
	return data->dipoleMoment;
}

bool Scan::magnet() const
{
	return data->magnet;
}

double Scan::drPower() const
{
	return data->drPower;
}

double Scan::pressure() const
{
	return data->pressure;
}

QStringList Scan::gasNames() const
{
	return data->gasNames;
}

QList<double> Scan::gasFlows() const
{
    return data->gasFlows;
}

double Scan::repRate() const
{
    return data->repRate;
}

QList<PulseGenerator::PulseChannelConfiguration> Scan::pulseConfiguration() const
{
	return data->pulseConfig;
}

int Scan::completedShots() const
{
	return data->completedShots;
}

int Scan::targetShots() const
{
	return data->targetShots;
}

bool Scan::isInitialized() const
{
	return data->initialized;
}

bool Scan::isAcquisitionComplete() const
{
	if(!isInitialized())
		return false;
	else
		return (data->completedShots >= data->targetShots) || data->aborted;
}

bool Scan::isSaved() const
{
	return data->saved;
}

bool Scan::isAborted() const
{
	return data->aborted;
}

bool Scan::isDummy() const
{
	return data->dummy;
}

bool Scan::skipTune() const
{
	return data->skipTune;
}

int Scan::tuningVoltage() const
{
    return data->tuningVoltage;
}

bool Scan::tuningVoltageTakenWithScan() const
{
    return data->tuningVoltageTakenWithScan;
}

int Scan::scansSinceTuningVoltageTaken() const
{
    return data->scansSinceTuningVoltageTaken;
}

int Scan::cavityVoltage() const
{
    return data->cavityVoltage;
}

int Scan::protectionDelayTime() const
{
    return data->protectionDelayTime;
}

int Scan::scopeDelayTime() const
{
    return data->scopeDelayTime;
}

void Scan::setNumber(int n)
{
	data->number = n;
}

void Scan::increment()
{
	data->completedShots++;
}

void Scan::setFid(const Fid f)
{
	data->fid = f;
}

void Scan::setProbeFreq(const double f)
{
	data->fid.setProbeFreq(f);
}

void Scan::setFtFreq(const double f)
{
	data->ftFreq = f;
}

void Scan::setDrFreq(const double f)
{
	data->drFreq = f;
}

void Scan::setAttenuation(const int a)
{
	data->ftAtten = a;
}

void Scan::setCavityVoltage(const int v)
{
   data->cavityVoltage = v;
}

void Scan::setProtectionDelayTime(const int v)
{
   data->protectionDelayTime = v;
}

void Scan::setScopeDelayTime (const int v)
{
   data->scopeDelayTime = v;
}

void Scan::setDipoleMoment( const double v)
{
    data->dipoleMoment = v;
}

void Scan::setDrPower(const double p)
{
	data->drPower = p;
}

void Scan::setPressure(const double p)
{
	data->pressure = p;
}

void Scan::setGasNames(const QStringList s)
{
	data->gasNames = s;
}

void Scan::setGasFlows(const QList<double> f)
{
	data->gasFlows = f;
}

void Scan::setTargetShots(int n)
{
    data->targetShots = n;
}

void Scan::setRepRate(const double rr)
{
    data->repRate = rr;
}

void Scan::setPulseConfiguration(const QList<PulseGenerator::PulseChannelConfiguration> p)
{
	data->pulseConfig = p;
}

void Scan::initializationComplete()
{
	data->ts = QDateTime::currentDateTime();
	data->initialized = true;
}

void Scan::save()
{

	//figure out scan number and where to save data
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	data->number = s.value(QString("scanNum"),0).toInt()+1;

	int dirMillionsNum = (int)floor((double) number()/1000000.0);
	int dirThousandsNum = (int)floor((double) number()/1000.0);

	QDir d(QString("/home/data/QtFTM/scans/%1/%2").arg(dirMillionsNum).arg(dirThousandsNum));
	if(!d.exists())
	{
		if(!d.mkpath(d.absolutePath()))
		{
			//this is bad... abort!
			data->aborted=true;
			return;
		}
	}

	//create output file
	QFile f(QString("%1/%2.txt").arg(d.absolutePath()).arg(number()));

	if(!f.open(QIODevice::WriteOnly))
	{
		//this is also bad... abort!
		data->aborted = true;
		return;
	}

	QTextStream t(&f);

	//write header and column heading
	t << scanHeader();
	t << QString("\nfid%1").arg(number());
	t.setRealNumberNotation(QTextStream::ScientificNotation);

	//this controls how many digits are printed after decimal.
	//in principle, an 8 bit digitizer requires only 3 digits (range = -127 to 128)
	//For each ~factor of 10 averages, we need ~one more digit of precision
	//This starts at 7 digits (sci notation; 6 places after decimal), and adds 1 for every factor of 10 shots.
	int logFactor = 0;
	if(completedShots() > 0)
		logFactor = (int)floor(log10((double)completedShots()));
	t.setRealNumberPrecision(6+logFactor);

	//write data
    for(int i=0; i<fid().size(); i++)
		t << QString("\n") << fid().at(i);

	t.flush();
	f.close();

	//increment scan number
	s.setValue(QString("scanNum"),data->number);
    s.sync();
	data->saved = true;
}

void Scan::abortScan()
{
	data->aborted = true;
}

void Scan::setDummy()
{
	data->dummy = true;
}

void Scan::setSkiptune(bool b)
{
	data->skipTune = b;
}

void Scan::setTuningVoltage(int v)
{
	data->tuningVoltage = v;
}

void Scan::setTuningVoltageTakenWithScan(bool yesOrNo)
{
    data->tuningVoltageTakenWithScan = yesOrNo;
}

void Scan::setScansSinceTuningVoltageTaken(int howMany)
{
    data->scansSinceTuningVoltageTaken = howMany;
}

void Scan::setMagnet(bool b)
{
	data->magnet = b;
}

QString Scan::scanHeader() const
{
	QString out;
	QTextStream t(&out);

    t.setRealNumberPrecision(12);
    QString yesOrNo;
    if(tuningVoltageTakenWithScan()) yesOrNo = QString("yes");
    else yesOrNo = QString("no");

	t << QString("#Scan\t") << number() << QString("\t\n");
	t << QString("#Date\t") << timeStamp().toString() << QString("\t\n");
	t << QString("#Shots\t") << completedShots() << QString("\t\n");
	t << QString("#Cavity freq\t") << ftFreq() << QString("\tMHz\n");
	t << QString("#Skipped Tuning\t") << skipTune() << QString("\t\n");
    t << QString("#Tuning Voltage\t") << tuningVoltage() << QString("\tmV\n");
    t << QString("#Was Tuning Voltage Taken with Scan\t") << yesOrNo << QString("\t\n");
    t << QString("#How Many Scans since tuning voltage was taken\t") << scansSinceTuningVoltageTaken() << QString("\t\n");
	t << QString("#Attenuation\t") << attenuation() << QString("\tdB\n");
    t << QString("#Dipole Moment\t") << dipoleMoment() << QString("\tD\n");
    t << QString("#Cavity Voltage\t") << cavityVoltage() << QString("\tmV\n");
    t << QString("#Protection Delay\t") << protectionDelayTime() << QString("\tus\n");
    t << QString("#Scope Delay\t") << scopeDelayTime() << QString("\tus\n");
    t << QString("#Magnet enabled\t") << magnet() << QString("\t\n");
//	t << QString("#DR enabled\t") << pulseConfiguration().at(3).enabled << QString("\t\n");
	t << QString("#DR freq\t") << drFreq() << QString("\tMHz\n");
	t << QString("#DR power\t") << drPower() << QString("\tdBm\n");
	t << QString("#Probe freq\t") << fid().probeFreq() << QString("\tMHz\n");
	t << QString("#FID spacing\t") << fid().spacing() << QString("\ts\n");
	t << QString("#FID points\t") << fid().size() << QString("\t\n");
    t << QString("#Rep rate\t") << repRate() << QString("\tHz\n");
	t << QString("#Pressure\t") << pressure() << QString("\tkTorr\n");
	for(int i=0; i<gasFlows().size(); i++)
	{
		t << QString("#Gas %1 name\t").arg(i+1);
		if(i < gasNames().size())
			t << gasNames().at(i);
		t << QString("\t\n#Gas %1 flow\t").arg(i+1) << gasFlows().at(i) << QString("\tsccm\n");
	}

	for(int i=0; i<pulseConfiguration().size(); i++)
	{
		QString prefix = QString("#Pulse ch %1 ").arg(pulseConfiguration().at(i).channel);
		t << prefix << QString("name\t") << pulseConfiguration().at(i).channelName << QString("\t\n");
		t << prefix << QString("active level\t") << pulseConfiguration().at(i).active;
		if(pulseConfiguration().at(i).active == PulseGenerator::ActiveHigh)
			t << QString("\tActive High\n");
		else
			t << QString("\tActive Low\n");
		t << prefix << QString("enabled\t") << pulseConfiguration().at(i).enabled << QString("\t\n");
		t << prefix << QString("delay\t") << pulseConfiguration().at(i).delay << QString("\tus\n");
		t << prefix << QString("width\t") << pulseConfiguration().at(i).width << QString("\tus\n");
	}

	t.flush();
	return out;
}

void Scan::parseFile(int num)
{
	if(num<1)
		return;

	int dirMillionsNum = (int)floor((double) num/1000000.0);
	int dirThousandsNum = (int)floor((double) num/1000.0);

	QFile f(QString("/home/data/QtFTM/scans/%1/%2/%3.txt").arg(dirMillionsNum).arg(dirThousandsNum).arg(num));

	if(!f.exists())
		return;

	if(!f.open(QIODevice::ReadOnly))
		return;

	QVector<double> fidData;

	while(!f.atEnd())
	{
		QString line(f.readLine());
		if(line.startsWith(QString("#FID points")))
			fidData.reserve(line.split(QChar(0x09)).at(1).toInt());
		else if(line.startsWith(QString("#")))
			parseFileLine(line);
		else if(line.startsWith(QString("fid")))
			break;
	}

	while(!f.atEnd())
		fidData.append(QString(f.readLine()).toDouble());

	data->fid.setData(fidData);
	data->saved = true;
	data->initialized = true;
	data->targetShots = completedShots();

}

void Scan::parseFileLine(QString s)
{
	QStringList sl = s.split(QChar(0x09));
	if(sl.size()<2)
		return;

	QString key = sl.at(0);
	QString val = sl.at(1);

    if((key.startsWith(QString("#Scan"))) && (key.endsWith(QString("can"))))
		data->number = val.toInt();
	else if(key.startsWith(QString("#Date")))
		data->ts = QDateTime::fromString(val);
	else if(key.startsWith(QString("#Shots")))
		data->completedShots = val.toInt();
	else if(key.startsWith(QString("#Cavity freq")))
		data->ftFreq = val.toDouble();
	else if(key.startsWith(QString("#Skipped")))
		data->skipTune = (bool)val.toInt();
    else if(key.startsWith(QString("#Tuning")))
        data->tuningVoltage = val.toInt();
    else if(key.startsWith(QString("#Was Tuning"))) {
        if(val == "yes")
        data->tuningVoltageTakenWithScan = 1;
        else data->tuningVoltageTakenWithScan = 0;
    }
    else if(key.startsWith(QString("#How Many ")))
        data->scansSinceTuningVoltageTaken = val.toInt();

    else if(key.startsWith(QString("#Cavity Voltage"), Qt::CaseInsensitive))
        data->cavityVoltage = val.toInt();
    else if(key.startsWith(QString("#Protection Delay"), Qt::CaseInsensitive))
        data->protectionDelayTime = val.toInt();
    else if(key.startsWith(QString("#Scope Delay"), Qt::CaseInsensitive))
        data->scopeDelayTime = val.toInt();
    else if(key.startsWith(QString("#Attenuation")))
        data->ftAtten = val.toInt();
    else if(key.startsWith(QString("#Dipole Moment")))
        data->dipoleMoment = val.toDouble();
	else if(key.startsWith(QString("#Magnet")))
		data->magnet = (bool)val.toInt();
	else if(key.startsWith(QString("#DR freq")))
		data->drFreq = val.toDouble();
	else if(key.startsWith(QString("#DR power")))
		data->drPower = val.toDouble();
	else if(key.startsWith(QString("#Pressure")))
		data->pressure = val.toDouble();
	else if(key.startsWith(QString("#Probe freq")))
		data->fid.setProbeFreq(val.toDouble());
	else if(key.startsWith(QString("#FID spacing")))
		data->fid.setSpacing(val.toDouble());
    else if(key.startsWith(QString("#Rep rate")))
        data->repRate = val.toDouble();
	else if(key.startsWith(QString("#Gas")))
	{
		if(key.endsWith(QString("name")))
			data->gasNames.append(val);
		else if(key.endsWith(QString("flow")))
			data->gasFlows.append(val.toDouble());
	}
	else if(key.startsWith(QString("#Pulse ch")))
	{
		int ch = key.split(QChar(0x20)).at(2).toInt();
		if(key.endsWith(QString("name")))
		{
			if(ch-1 >= data->pulseConfig.size())
			{
				data->pulseConfig.append(PulseGenerator::PulseChannelConfiguration());
				data->pulseConfig[ch-1].channel = ch;
			}
			data->pulseConfig[ch-1].channelName = val;
		}
		else if(key.endsWith(QString("level")))
		{
			if(ch-1 >= data->pulseConfig.size())
			{
				data->pulseConfig.append(PulseGenerator::PulseChannelConfiguration());
				data->pulseConfig[ch-1].channel = ch;
			}
			data->pulseConfig[ch-1].active = (PulseGenerator::ActiveLevel)val.toInt();
		}
		else if(key.endsWith(QString("enabled")))
		{
			if(ch-1 >= data->pulseConfig.size())
			{
				data->pulseConfig.append(PulseGenerator::PulseChannelConfiguration());
				data->pulseConfig[ch-1].channel = ch;
			}
			data->pulseConfig[ch-1].enabled = (bool)val.toInt();
		}
		else if(key.endsWith(QString("delay")))
		{
			if(ch-1 >= data->pulseConfig.size())
			{
				data->pulseConfig.append(PulseGenerator::PulseChannelConfiguration());
				data->pulseConfig[ch-1].channel = ch;
			}
			data->pulseConfig[ch-1].delay = val.toDouble();
		}
		else if(key.endsWith(QString("width")))
		{
			if(ch-1 >= data->pulseConfig.size())
			{
				data->pulseConfig.append(PulseGenerator::PulseChannelConfiguration());
				data->pulseConfig[ch-1].channel = ch;
			}
			data->pulseConfig[ch-1].width = val.toDouble();
		}
	}
}