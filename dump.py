import os
import pandas as pd
from src.services.oracle_db_service import oracle_db_service
from src.state.store import InMemoryStore, get_session, set_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DmrPartialReleaseAutomationService:
    """Service workflows for DMR Partial Release resolution on Oracle DB."""

    async def triage_dmr_partial_release(
        self,
        thread_id: str,
        store: InMemoryStore,
    ) -> dict:
        # Step 1: Check for and read the uploaded CSV file (cassette list)
        file_path = os.path.join("uploads", f"{thread_id}_dmr.csv")
        if not os.path.exists(file_path):
            return {
                "next_action": (
                    "Tell the user only this: You have not uploaded the CSV file for this incident which contains the cassette list. "
                    "Please upload the CSV file before I can proceed with the partial release of DMR.\n"
                    "Once the user confirms they have uploaded the file, proceed calling triage_dmr_partial_release tool again."
                )
            }
        
        df = pd.read_csv(file_path)
        if 'mcass_id' in df.columns:
            cassette_ids = df['mcass_id'].astype(str).tolist()
        elif 'cassette_id' in df.columns:
            cassette_ids = df['cassette_id'].astype(str).tolist()
        else:
            cassette_ids = df.iloc[:, 0].astype(str).tolist()
        
        if not cassette_ids:
            return {"next_action": "Tell the user only this: The uploaded CSV file is empty. Please upload a valid CSV file with cassette IDs."}

        # Step 2: Confirm DMR / Hold status
        # 2A: Get DMR from hold_rels_detail for the cassettes from the CSV.
        cassette_ids_str = ', '.join([f"'{cid}'" for cid in cassette_ids])
        sql_2a = f"SELECT DISTINCT mcass_id, dmr_no FROM hold_rels_detail WHERE mcass_id IN ({cassette_ids_str})"
        hold_details = await oracle_db_service.fetch_all(sql_2a)
        
        if not hold_details:
            return {"next_action": "Tell the user only this: No hold records were found in the Oracle DB for the provided cassettes. So we cannot move forward to do partial release of these DMRs."}

        # Group cassettes by DMR
        dmr_to_cassettes = {}
        for detail in hold_details:
            dmr = detail.get('dmr_no')
            cassette = detail.get('mcass_id')
            if dmr:
                if dmr not in dmr_to_cassettes:
                    dmr_to_cassettes[dmr] = []
                dmr_to_cassettes[dmr].append(cassette)

        eligible_dmrs = []
        rejected_info = []

        # 2B & 2C: Check hold status + flags for EACH DMR
        for dmr_no in dmr_to_cassettes.keys():
            sql_2bc = "SELECT hold_status, drb_flag, yield_flag FROM p_hold_rels_log WHERE dmr_no = :dmr"
            rels_log = await oracle_db_service.fetch_one(sql_2bc, {"dmr": dmr_no})
            
            if not rels_log:
                rejected_info.append(f"DMR {dmr_no}: No entry found in p_hold_rels_log.")
                continue

            hold_status = rels_log.get('hold_status', '')
            drb_flag = rels_log.get('drb_flag', '')
            yield_flag = rels_log.get('yield_flag', '')

            # Decision A: Is hold_status = TATEST / TTATEST / TTTATEST?
            if hold_status not in ('TATEST', 'TTATEST', 'TTTATEST'):
                rejected_info.append(f"DMR {dmr_no}: Ineligible hold status ({hold_status}).")
                continue

            # Decision B: Are drb_flag = 'T' AND yield_flag = 'T'?
            if drb_flag == 'T' and yield_flag == 'T':
                eligible_dmrs.append({
                    "dmr_no": dmr_no,
                    "hold_status": hold_status,
                    "drb_flag": drb_flag,
                    "yield_flag": yield_flag,
                    "cassettes": dmr_to_cassettes[dmr_no]
                })
            elif yield_flag == 'CO':
                rejected_info.append(f"DMR {dmr_no}: Yield flag is CO.")
            else:
                rejected_info.append(f"DMR {dmr_no}: Flags not ready (DRB: {drb_flag}, Yield: {yield_flag}).")

        session = await get_session(store, thread_id)
        
        if rejected_info:
            email_draft = "Subject: DMR Partial Release Ineligibility Notification\n\nHello,\n\nPlease be advised that partial release was not possible for the following DMRs due to the reasons listed below:\n\n"
            for info in rejected_info:
                email_draft += f"- {info}\n"
            email_draft += "\nPlease review these DMRs.\n"
            session["email_draft"] = email_draft
        
        if not eligible_dmrs:
            await set_session(store, thread_id, session)
            return {
                "next_action": "Tell the user only this: We reviewed all the provided DMRs, and none of them were eligible for partial release."
            }

        # Store eligible data in session
        session["eligible_dmrs"] = eligible_dmrs
        await set_session(store, thread_id, session)

        # Prepare summary for user
        summary = "I have analyzed the cassettes for DMR partial release and found the following DMRs that are eligible for partial release:\n\n"
        for item in eligible_dmrs:
            summary += f"- DMR: {item['dmr_no']} ({len(item['cassettes'])} cassettes)\n"
            summary += f"  Status: {item['hold_status']}, Flags: DRB={item['drb_flag']}, Yield={item['yield_flag']}\n"

        summary += "\nDo you want me to proceed with executing the partial release resolution?"
        
        return {
            "next_action": f"Tell the user only this:\n{summary}\n\nOnce the user confirms, proceed calling perform_dmr_partial_release_resolution tool."
        }

    async def execute_dmr_partial_release(
        self,
        thread_id: str,
        store: InMemoryStore,
        operator_confirmed: bool = False,
    ) -> dict:
        if not operator_confirmed:
            return {"next_action": "Tell the user only this: Operator confirmation is required before execution."}

        session = await get_session(store, thread_id)
        eligible_dmrs = session.get("eligible_dmrs", [])
        if not eligible_dmrs:
            return {"next_action": "Tell the user only this: No eligible DMRs found in session. Please run triage again."}

        results = []
        skipped_dmrs = []
        for item in eligible_dmrs:
            dmr_no = item['dmr_no']
            cassette_ids = item['cassettes']
            
            # 3.1 & 3.2: Get details
            cassette_ids_str = ', '.join([f"'{cid}'" for cid in cassette_ids])
            sql_31 = f"SELECT DISTINCT mcass_id, dst_cid, dst_loc, lot_no FROM mdw_src_dest WHERE mcass_id IN ({cassette_ids_str})"
            tm1_data = await oracle_db_service.fetch_all(sql_31)
            
            eligible_cassette_ids = list(set([row['mcass_id'] for row in tm1_data if row.get('mcass_id')])) if tm1_data else []
            if not eligible_cassette_ids:
                skipped_dmrs.append(dmr_no)
                continue

            eligible_cassette_ids_str = ', '.join([f"'{cid}'" for cid in eligible_cassette_ids])
            
            # 3.3 Check if partial release already exists
            sql_33 = f"SELECT * FROM p_rels_log_dtl WHERE fg_id IN ({eligible_cassette_ids_str})"
            existing_rels = await oracle_db_service.fetch_all(sql_33)
            
            # 3.5 Create tem10 (template row)
            sql_35 = "SELECT * FROM p_rels_log_dtl WHERE date_tm > sysdate - 30 AND released = '1' AND rownum = 1"
            template_row = await oracle_db_service.fetch_one(sql_35)

            # 3.6 Insert into p_rels_log_dtl (COMMENTED OUT AS REQUESTED)
            # logger.info(f"WOULD INSERT release records for DMR {dmr_no} using template {template_row}")
            
            # 3.7 Delete from hold_rels_detail (COMMENTED OUT AS REQUESTED)
            # logger.info(f"WOULD DELETE hold records for cassettes {eligible_cassette_ids} under DMR {dmr_no}")
            
            results.append(f"Processed DMR {dmr_no}: {len(eligible_cassette_ids)} cassettes identified, {len(existing_rels)} existing releases found.")

        if skipped_dmrs:
            dmrs_str = ", ".join(skipped_dmrs)
            results.append(f"We were not able to process these DMRs ({dmrs_str}) because no cassettes were found in the mdw_src_dest table.")

        return {
            "next_action": (
                f"Tell the user only this: Partial release resolution has been processed (Simulation Mode).\n\n" +
                "\n".join(results) +
                "\n\nNote: Actual database writes (Insert/Delete) were skipped as per configuration."
            )
        }

dmr_partial_release_automation_service = DmrPartialReleaseAutomationService()
