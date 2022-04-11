
select 
    title,
    username,
    displayname,
    versionnumber,
    totallines,
    competitionTitle,
    competitionSubTitle
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
                    ((((kernelversioncompetitionsources kvc inner join kernelversions kv on kvc.kernelversionid = kv.id) 
                    inner join kernels k on k.CurrentKernelVersionId = kv.id) 
                    inner join competitions c on kvc.sourcecompetitionid = c.id)
                    inner join users u on u.id = k.authoruserid)
                where
                    kv.title not like '%tutorial%' and 
                    kv.title not like '%guide%' and
                    kv.title not like '%beginner%' and
                    kv.title not like '%introduction%'
                order by 
                    k.totalvotes DESC
                limit 13675
            ) as tb
        where
            tb.totallines > 399 and
            tb.versionnumber > 9
    ) as tbtb
where 
	tbtb.competitionTitle like '%image%' or tbtb.competitionTitle like '%3d%' or tbtb.competitionTitle like '%vision%' or 
	tbtb.competitionSubTitle like '%image%' or tbtb.competitionSubTitle like '%3d%' or tbtb.competitionSubTitle like '%vision%' or 
	tbtb.title like '%image%' or tbtb.title like '%3d%' or tbtb.title like '%vision%'
order by
    tbtb.versionnumber ASC
limit 200;