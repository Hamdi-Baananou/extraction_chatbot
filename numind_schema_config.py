"""
NuMind Schema Configuration

This file contains the extraction schema that matches your NuMind playground configuration.
Update this schema to match exactly what you have configured in your NuMind project.

To get your schema from NuMind playground:
1. Go to your NuMind project
2. Look at the extraction schema configuration
3. Copy it here and replace the default schema below
"""

# Your exact NuMind template converted to JSON schema format
CUSTOM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "Material Filling": {
            "type": "string",
            "description": "The material filling or additive used in the connector",
            "enum": ["none", "GF", "CF", "(GB+GF)"]
        },
        "Material Name": {
            "type": "string", 
            "description": "The main material name of the connector",
            "enum": ["PA66", "PBT", "PA", "Silicone Rubber", "PA6", "Plastics", "PP", "PA+SPS", "PA12", "PET", "PA66+PA6", "PC"]
        },
        "Max. Working Temperature [°C]": {
            "type": "string",
            "description": "Maximum working temperature in Celsius",
            "enum": ["40.0000", "80.0000", "85.0000", "100.000", "105.000", "120.000", "125.000", "130.000", "135", "140.000", "150.000", "155.000", "-1"]
        },
        "Min. Working Temperature [°C]": {
            "type": "string",
            "description": "Minimum working temperature in Celsius",
            "enum": ["-65.0000", "-55.0000", "-40.0000", "-30.0000", "-20.0000", "-1"]
        },
        "Colour": {
            "type": "string",
            "description": "Color of the connector",
            "enum": ["000 bk", "101 nt", "111 ye", "222 og", "333 rd", "353 pk", "444 vt", "555 bu", "666 gn", "777 gy", "888 bn", "999 wh"]
        },
        "Contact Systems": {
            "type": "string",
            "description": "Contact system type",
            "enum": ["TAB 1.8", "0.64", "MCP 2.8", "MLK 1.2", "MQS 0.64", "SLK 2.8", "HF", "070", "GT 2.8", "MTS 0.64", "NG 1.8", "2.3", "BOX 2.8", "QKK 2.8", "RH 0.64", "CTS 1.5", "NanoMQS", "MCON 1.2", "HSD", "RK", "YESC 1.5", "MCP 1.5K", "HCT4", "HPCS 2.8", "2.8", "040", "SPT 4.8", "090 HW", "AMPSEAL", "MOD", "ST", "CONI1 1.6", "Econoseal 1.5", "MCP 1.2", "TAB 1.2", "FASTON 6.3", "M800", "GET 0.64", "MATE-N-LOK", "025 TH", "MPQ 2.8", "MAK 8", "MAK 2.8", "TAB 1.5", "DIA 3.6", "DIA 9.0", "DIA 6.0", "DIA 3.0", "TAB 1.6", "QKK 4.8", "FS 2.8", "FS 1.2", "US 2.8x0.8", "TAB 2.8", "TAB 4.8", "TAB 9.5", "3.5", "MCP 6.3", "MX 1.5", "1.5", "1.2", "QKK 1.2", "MLK 1.2 Sm", "MCP 1.5", "MQS 1.5", "MQS 0.64 CB"]
        },
        "Gender": {
            "type": "string",
            "description": "The gender of the connector (male/female)",
            "enum": ["female", "male"]
        },
        "Housing Seal": {
            "type": "string",
            "description": "Housing seal information",
            "enum": ["none", "interface seal", "radial seal"]
        },
        "HV Qualified": {
            "type": "string",
            "description": "High voltage qualification information",
            "enum": ["No", "Yes"]
        },
        "Mechanical Coding": {
            "type": "string",
            "description": "Mechanical coding or keying information",
            "enum": ["None", "A", "B", "C", "D", "E", "F", "G", "I", "Z", "1", "2", "5", "III", "No naming", "Neutral", "X", "II", "V"]
        },
        "Number Of Cavities": {
            "type": "string",
            "description": "Number of cavities or positions in the connector",
            "enum": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "14", "16", "18", "19", "20", "23", "24", "26", "27", "30", "31", "32", "35", "38", "46", "47", "52", "53", "60", "63", "64", "136"]
        },
        "Number Of Rows": {
            "type": "string",
            "description": "Number of rows in the connector",
            "enum": ["0", "1", "2", "4", "7", "9", "24"]
        },
        "Pre-assembled": {
            "type": "string",
            "description": "Pre-assembly information",
            "enum": ["No", "Yes"]
        },
        "Sealing": {
            "type": "string",
            "description": "Sealing information",
            "enum": ["unsealed", "sealed"]
        },
        "Sealing Class": {
            "type": "string",
            "description": "Sealing class or IP rating",
            "enum": ["IPx0", "IPx7", "IPx9K", "IPx6", "IPx4", "IPx8", "IPx5", "not defined", "IPx9K,IPx6", "IPx9K,IPx7", "IPx9K,IPx9K", "IPx6,IPx7", "IPx7,IPx9K", "IPx7,IPx6"]
        },
        "Terminal Position Assurance": {
            "type": "string",
            "description": "Terminal position assurance information",
            "enum": ["None", "1", "2", "undefined_to do not use"]
        },
        "Type Of Connector": {
            "type": "string",
            "description": "Type of connector",
            "enum": ["Standard", "Antenna", "Contact Carrier", "HSD / USB / HDMI", "Airbag / Squib", "IDC", "Bulb holder", "Relay holder"]
        },
        "Wire Seal": {
            "type": "string",
            "description": "Wire seal information",
            "enum": ["none", "single wire seal", "Mat seal", "Silicone family seal", "family seal"]
        },
        "Connector Position Assurance": {
            "type": "string",
            "description": "Connector position assurance information",
            "enum": ["No", "Yes"]
        },
        "Colour Coding": {
            "type": "string",
            "description": "Color coding information",
            "enum": ["None", "Red", "Blue", "Orange", "Natural", "Black", "Pink", "White", "Violet"]
        },
        "Set/Kit": {
            "type": "string",
            "description": "Set or kit information",
            "enum": ["No", "Yes"]
        },
        "Name Of Closed Cavities": {
            "type": "string",
            "description": "Information about closed cavities",
            "enum": ["none", "2,3", "4-7,14-17", "4-5,10,14-15,17,19"]
        },
        "Pull-To-Seat": {
            "type": "string",
            "description": "Pull-to-seat force or mechanism information",
            "enum": ["No", "Yes"]
        }
    },
    "required": []
}

def get_custom_schema():
    """Returns the custom extraction schema."""
    return CUSTOM_EXTRACTION_SCHEMA 