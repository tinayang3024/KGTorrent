
select *
from
    (
        select *
        from 
            (
                select
                    kv.title, 
                    k. CurrentUrlSlug,
                    u.username as UserName, 
                    k.CurrentKernelVersionId,
                    u.displayname, 
                    kv.versionnumber, 
                    kv.totallines, 
                    c.title as competitionTitle, 
                    c.subtitle as competitionSubTitle
                from 
                    (((((kernelversioncompetitionsources kvc inner join kernelversions kv on kvc.kernelversionid = kv.id) 
                    inner join kernels k on k.CurrentKernelVersionId = kv.id) 
                    inner join competitions c on kvc.sourcecompetitionid = c.id)
                    inner join users u on u.id = k.authoruserid)
                    inner join kernellanguages kl on kv.scriptlanguageid = kl.id)
                where
                    kv.title not like '%tutorial%' and 
                    kv.title not like '%guide%' and
                    kv.title not like '%beginner%' and
                    kv.title not like '%introduction%' and
                    kl.name like 'IPython Notebook HTML'
                order by 
                    k.totalvotes DESC
                limit 13675
            ) as tb
        where
            tb.totallines > 399 and
            tb.versionnumber > 9
    ) as tbtb
where 
	tbtb.competitionTitle like '%sentiment%' or tbtb.competitionTitle like '%attitude%' or 
	tbtb.competitionSubTitle like '%sentiment%' or tbtb.competitionSubTitle like '%attitude%' or
	tbtb.title like '%sentiment%' or tbtb.title like '%attitude%' 
limit 200;