from peewee import *
from playhouse.postgres_ext import *

from datetime import datetime, timezone, timedelta

POSTGRESQL_SETTINGS = {
       'DATABASE': 'Your database name',
       'USER': 'your superuser name?',
       'HOST': 'Postgresql IP',
       'PORT': 'Postgresql Port',
       'PASSWORD': "Super secret password here"
   }

postgres_database = PostgresqlExtDatabase(
    POSTGRESQL_SETTINGS['DATABASE'],
    user=POSTGRESQL_SETTINGS['USER'],
    host=POSTGRESQL_SETTINGS['HOST'],
    port=POSTGRESQL_SETTINGS['PORT'],
    password=POSTGRESQL_SETTINGS['PASSWORD'],
    register_hstore=False, autocommit=False, autorollback=False)

class BaseModel(Model):
    class Meta:
        database = postgres_database


class Author(BaseModel):
    authorID = IntegerField(primary_key=True)
    name = CharField()
    semanticScholarUrl = CharField()

    hIndex = IntegerField(default=0)
    influentialCitationCount = IntegerField(default=0)
    citationVelocity = IntegerField(default=0)
    totalInfluentialCitationCount = IntegerField(default=0)
    maxEstCitationAcceleration = DoubleField(default=0.0)
    minEstCitationAcceleration = DoubleField(default=0.0)
    estCitationAcceleration = DoubleField(default=0.0)
    estCitationAccelerationConfidence = DoubleField(default=0.0)

    influencedIDList = ArrayField(CharField, default=list)
    influencedPaper = JSONField(default=dict)
    influenceCount = IntegerField(default=0)

    influencedByIDList = ArrayField(CharField, default=list)
    influencedByPaper = JSONField(default=dict)    
    influenceByCount = IntegerField(default=0)

    citationHistory = JSONField(default=dict)

    totalPaper = IntegerField(default=0)

    # paperHistory = JSONField(default=dict)

    class Meta:
        db_table = 'fyp_author'


class Paper(BaseModel):
    indexID = IntegerField(default=0)
    arvixID = CharField(primary_key=True)
    # semantic scholar id
    paperId = CharField(default='None')
    doiID = CharField(default='None')

    title = TextField()
    summary = TextField()
    affiliation = ArrayField(CharField, default=list)
    category = ArrayField(CharField, default=list) # term
    comments = TextField()
    journal_ref = TextField()
    
    # semantic scholar author id
    url = CharField()
    authorID = ArrayField(IntegerField, default=list)
    authorName = ArrayField(CharField, default=list)
    authorCount = IntegerField(default=0)
    # submissionTo = CharField()

    # meta data
    pages = IntegerField(default=-1)
    figures = IntegerField(default=-1)
    table = IntegerField(default=-1)

    publishedDate = DateTimeField()

    # semantic scholar stats
    citationVelocity = IntegerField(default=0)
 
    referencesCount = IntegerField(default=0)
    topics = JSONField(default=dict)
    keywords = ArrayField(CharField, default=list)
    venue = CharField(default='None')
    year = IntegerField(default=0)
    influentialCitationCount = IntegerField(default=0)
    citations = JSONField(default=dict)
    # citationCount : target!!!!!!!!!!!
    citationCount = IntegerField(default=0)

    class Meta:
        db_table = 'fyp_paper'


def create_table():
    postgres_database.connect()
    postgres_database.create_tables([Author])