from django.db import models
import datetime
class labourMaster(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "labour_master"
      

class labourMasterAggregate(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "labour_master_aggregate"
     

class materialMaster(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "material_master"
  

class materialMasterAggregate(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "material_master_aggregate"
    


class completionRate(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    completion_rate = models.CharField(max_length = 100, null = True, blank= True)
    class Meta:
        db_table = "completion_rate"

class purchaseYearlyTotal(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "purchase_yearly_total"

class purchaseAggregate(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "purchase_aggregate"

class masterYearlyTotal(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    master_total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "master_yearly_total"

class masterAggregate(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    master_total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "master_aggregate"

class requiredTable(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    labour_cost = models.FloatField(null=True, blank=True)
    material_cost = models.FloatField(null=True, blank=True)
    purchase_orders = models.FloatField(null=True, blank=True)
    total_nrc = models.FloatField(null=True, blank=True)
    completion_rate = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "required_table"

class AITableOne(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    master_total = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "ai_table_one"


class AITrainerData(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january_po = models.FloatField(null=True, blank=True)
    february_po = models.FloatField(null=True, blank=True)
    march_po = models.FloatField(null=True, blank=True)
    april_po = models.FloatField(null=True, blank=True)
    may_po = models.FloatField(null=True, blank=True)
    june_po = models.FloatField(null=True, blank=True)
    july_po = models.FloatField(null=True, blank=True)
    august_po = models.FloatField(null=True, blank=True)
    september_po = models.FloatField(null=True, blank=True)
    october_po = models.FloatField(null=True, blank=True)
    november_po = models.FloatField(null=True, blank=True)
    december_po = models.FloatField(null=True, blank=True)
    january_lb = models.FloatField(null=True, blank=True)
    february_lb = models.FloatField(null=True, blank=True)
    march_lb = models.FloatField(null=True, blank=True)
    april_lb = models.FloatField(null=True, blank=True)
    may_lb = models.FloatField(null=True, blank=True)
    june_lb = models.FloatField(null=True, blank=True)
    july_lb = models.FloatField(null=True, blank=True)
    august_lb = models.FloatField(null=True, blank=True)
    september_lb = models.FloatField(null=True, blank=True)
    october_lb = models.FloatField(null=True, blank=True)
    november_lb = models.FloatField(null=True, blank=True)
    december_lb = models.FloatField(null=True, blank=True)
    january_mt = models.FloatField(null=True, blank=True)
    february_mt = models.FloatField(null=True, blank=True)
    march_mt = models.FloatField(null=True, blank=True)
    april_mt = models.FloatField(null=True, blank=True)
    may_mt = models.FloatField(null=True, blank=True)
    june_mt = models.FloatField(null=True, blank=True)
    july_mt = models.FloatField(null=True, blank=True)
    august_mt = models.FloatField(null=True, blank=True)
    september_mt = models.FloatField(null=True, blank=True)
    october_mt = models.FloatField(null=True, blank=True)
    november_mt = models.FloatField(null=True, blank=True)
    december_mt = models.FloatField(null=True, blank=True)
    completion_rate = models.FloatField(null=True, blank=True)
    master_total =  models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "ai_trainer_data"
    
class AITrainerDataScaled(models.Model):
    system_code = models.CharField(max_length = 100, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    january_po = models.FloatField(null=True, blank=True)
    february_po = models.FloatField(null=True, blank=True)
    march_po = models.FloatField(null=True, blank=True)
    april_po = models.FloatField(null=True, blank=True)
    may_po = models.FloatField(null=True, blank=True)
    june_po = models.FloatField(null=True, blank=True)
    july_po = models.FloatField(null=True, blank=True)
    august_po = models.FloatField(null=True, blank=True)
    september_po = models.FloatField(null=True, blank=True)
    october_po = models.FloatField(null=True, blank=True)
    november_po = models.FloatField(null=True, blank=True)
    december_po = models.FloatField(null=True, blank=True)
    january_lb = models.FloatField(null=True, blank=True)
    february_lb = models.FloatField(null=True, blank=True)
    march_lb = models.FloatField(null=True, blank=True)
    april_lb = models.FloatField(null=True, blank=True)
    may_lb = models.FloatField(null=True, blank=True)
    june_lb = models.FloatField(null=True, blank=True)
    july_lb = models.FloatField(null=True, blank=True)
    august_lb = models.FloatField(null=True, blank=True)
    september_lb = models.FloatField(null=True, blank=True)
    october_lb = models.FloatField(null=True, blank=True)
    november_lb = models.FloatField(null=True, blank=True)
    december_lb = models.FloatField(null=True, blank=True)
    january_mt = models.FloatField(null=True, blank=True)
    february_mt = models.FloatField(null=True, blank=True)
    march_mt = models.FloatField(null=True, blank=True)
    april_mt = models.FloatField(null=True, blank=True)
    may_mt = models.FloatField(null=True, blank=True)
    june_mt = models.FloatField(null=True, blank=True)
    july_mt = models.FloatField(null=True, blank=True)
    august_mt = models.FloatField(null=True, blank=True)
    september_mt = models.FloatField(null=True, blank=True)
    october_mt = models.FloatField(null=True, blank=True)
    november_mt = models.FloatField(null=True, blank=True)
    december_mt = models.FloatField(null=True, blank=True)
    completion_rate = models.FloatField(null=True, blank=True)
    master_total =  models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "ai_trainer_data_scaled"

class graphData(models.Model):
    index = models.PositiveIntegerField(null = True, blank = True)
    label = models.CharField(max_length = 200, null = True, blank= True)
    january = models.FloatField(null=True, blank=True)
    february = models.FloatField(null=True, blank=True)
    march = models.FloatField(null=True, blank=True)
    april = models.FloatField(null=True, blank=True)
    may = models.FloatField(null=True, blank=True)
    june = models.FloatField(null=True, blank=True)
    july = models.FloatField(null=True, blank=True)
    august = models.FloatField(null=True, blank=True)
    september = models.FloatField(null=True, blank=True)
    october = models.FloatField(null=True, blank=True)
    november = models.FloatField(null=True, blank=True)
    december = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = 'graph_data'

class yearlyGraphData(models.Model):
    index = models.PositiveIntegerField(null = True, blank = True)
    systemcode = models.CharField(max_length = 200, null = True, blank= True)
    year = models.IntegerField(null = True, blank =True)
    total = models.FloatField(null=True, blank=True)
    label = models.CharField(max_length = 200, null = True, blank= True)
    class Meta:
        db_table = 'yearly_graph_data'


class accuracyTolerance(models.Model):
    accuracy = models.FloatField(null=True, blank=True)
    tolerance = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = 'accuracy_tolerance'

  